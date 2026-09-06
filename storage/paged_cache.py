"""Physical storage is shared; each request owns a logical block table."""
import math
import os
import torch


class BlockPool:
    def __init__(self, n_layers, n_blocks, block_size, n_kv_heads, d_head, *, device, dtype):
        if min(n_layers, n_blocks, block_size, n_kv_heads, d_head) <= 0:
            raise ValueError("pool dimensions must be positive")
        self.block_size = block_size
        self.n_blocks = n_blocks
        shape = (n_layers, n_blocks, n_kv_heads, block_size, d_head)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty_like(self.k)
        self.free = list(reversed(range(n_blocks)))
        self.owners = {}
        # Bound temporary attention storage independently of sequence length.
        # Setting this to block_size reproduces the original one-page loop.
        self.attention_tile_tokens = int(os.environ.get("NANOVLLM_ATTENTION_TILE_TOKENS", "1024"))
        if self.attention_tile_tokens < 1:
            raise ValueError("attention tile size must be positive")

    def allocate(self, owner, count):
        if count > len(self.free):
            raise MemoryError("KV block pool exhausted")
        blocks = [self.free.pop() for _ in range(count)]
        for block in blocks:
            self.owners[block] = {owner}
        return blocks

    def release(self, owner, blocks):
        if len(set(blocks)) != len(blocks) or any(owner not in self.owners.get(b, set()) for b in blocks):
            raise ValueError("invalid block ownership")
        for block in blocks:
            self.owners[block].remove(owner)
            if not self.owners[block]:
                del self.owners[block]
                self.free.append(block)

    def retain(self, owner, blocks):
        if any(b not in self.owners or owner in self.owners[b] for b in blocks):
            raise ValueError("invalid shared block reference")
        for block in blocks:
            self.owners[block].add(owner)

    @property
    def allocated_bytes(self):
        return 2 * self.k.numel() * self.k.element_size()

    @property
    def bytes_per_block(self):
        return self.allocated_bytes // self.n_blocks


class PagedCache:
    def __init__(self, pool):
        self.pool = pool
        self.block_table = []
        self.layers = [PagedLayer(self, i) for i in range(pool.k.shape[0])]
        self.closed = False
        self._table_key = None
        self._device_table = None

    def device_table(self):
        key = tuple(self.block_table)
        if self._table_key != key:
            self._device_table = torch.tensor(key, device=self.pool.k.device, dtype=torch.long)
            self._table_key = key
        return self._device_table

    @property
    def length(self):
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise RuntimeError("layer lengths disagree")
        return self.layers[0].length

    def reserve(self, end):
        if self.closed:
            raise ValueError("cache is closed")
        needed = math.ceil(end / self.pool.block_size)
        start = min(layer.length for layer in self.layers)
        cow = [i for i, b in enumerate(self.block_table) if len(self.pool.owners[b]) > 1
               and i * self.pool.block_size < end and (i + 1) * self.pool.block_size > start]
        extra = max(0, needed - len(self.block_table))
        allocated = self.pool.allocate(self, len(cow) + extra)
        try:
            for i, new in zip(cow, allocated):
                old = self.block_table[i]
                self.pool.k[:, new].copy_(self.pool.k[:, old])
                self.pool.v[:, new].copy_(self.pool.v[:, old])
        except Exception:
            self.pool.release(self, allocated)
            raise
        for i, new in zip(cow, allocated):
            self.pool.release(self, [self.block_table[i]])
            self.block_table[i] = new
        self.block_table.extend(allocated[len(cow):])

    def fork(self, length=None):
        if self.closed:
            raise ValueError("cache is closed")
        length = self.length if length is None else length
        if not 0 <= length <= self.length:
            raise ValueError("invalid shared prefix length")
        child = PagedCache(self.pool)
        child.block_table = self.block_table[:math.ceil(length / self.pool.block_size)]
        self.pool.retain(child, child.block_table)
        for layer in child.layers:
            layer.length = length
        return child

    def truncate(self, length):
        if not 0 <= length <= min(layer.length for layer in self.layers):
            raise ValueError("invalid rollback length")
        keep = math.ceil(length / self.pool.block_size)
        self.pool.release(self, self.block_table[keep:])
        del self.block_table[keep:]
        for layer in self.layers:
            layer.length = length

    def close(self):
        if not self.closed:
            self.pool.release(self, self.block_table)
            self.block_table.clear()
            for layer in self.layers:
                layer.length = 0
            self.closed = True

    @property
    def live_bytes(self):
        return self.length * self.pool.bytes_per_block // self.pool.block_size

    @property
    def assigned_bytes(self):
        return len(self.block_table) * self.pool.bytes_per_block


class PagedLayer:
    def __init__(self, cache, index):
        self.cache = cache
        self.index = index
        self.length = 0

    def append(self, k, v):
        pool = self.cache.pool
        if k.shape[0] != 1 or k.shape != v.shape:
            raise ValueError("paged layers accept a single request at a time")
        if k.dtype != pool.k.dtype or k.device != pool.k.device:
            raise ValueError("cache dtype/device mismatch")
        start, end = self.length, self.length + k.shape[2]
        self.cache.reserve(end)
        # Copy only new token slices, never concatenate the history.
        while start < end:
            logical, offset = divmod(start, pool.block_size)
            count = min(end - start, pool.block_size - offset)
            physical = self.cache.block_table[logical]
            source = start - self.length
            pool.k[self.index, physical, :, offset:offset + count].copy_(k[0, :, source:source + count])
            pool.v[self.index, physical, :, offset:offset + count].copy_(v[0, :, source:source + count])
            start += count
        self.length = end

    def attend(self, q, positions):
        """Online softmax over physical blocks, with no full-history gather.

        q is [1, Hq, T, Dh]. Accumulators retain a separate result per query.
        Grouped matmul reuses compact KV heads without repeat_interleave.
        """
        pool = self.cache.pool
        _, hq, t, d = q.shape
        hkv = pool.k.shape[2]
        groups = hq // hkv
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        query = q[0].reshape(hkv, groups, t, d).to(dtype)
        maximum = torch.full((hkv, groups, t, 1), -torch.inf, device=q.device, dtype=dtype)
        denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros((hkv, groups, t, d), device=q.device, dtype=dtype)
        pages_per_tile = max(1, pool.attention_tile_tokens // pool.block_size)
        table = self.cache.device_table()
        for logical in range(0, len(self.cache.block_table), pages_per_tile):
            start = logical * pool.block_size
            page_count = min(pages_per_tile, len(self.cache.block_table) - logical)
            count = min(page_count * pool.block_size, self.length - start)
            if count <= 0:
                break
            if page_count == 1:
                physical = self.cache.block_table[logical]
                k = pool.k[self.index, physical, :, :count].to(dtype)
                v = pool.v[self.index, physical, :, :count].to(dtype)
            else:
                physical = table[logical:logical + page_count]
                k = pool.k[self.index].index_select(0, physical).permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :count].to(dtype)
                v = pool.v[self.index].index_select(0, physical).permute(1, 0, 2, 3).reshape(hkv, -1, d)[:, :count].to(dtype)
            scores = query @ k[:, None].transpose(-2, -1) / math.sqrt(d)
            visible = torch.arange(start, start + count, device=q.device)[None, :] <= positions[:, None]
            scores = scores.masked_fill(~visible, -torch.inf)
            new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            # First block always contains position 0, visible to every query.
            weights = torch.exp(scores - new_max)
            correction = torch.exp(maximum - new_max)
            accumulator = accumulator * correction + weights @ v[:, None]
            denominator = denominator * correction + weights.sum(-1, keepdim=True)
            maximum = new_max
        return (accumulator / denominator).reshape(1, hq, t, d).to(q.dtype)

    def gather(self):
        """Dense oracle for tests only; the attention path never calls this."""
        pool = self.cache.pool
        keys, values = [], []
        for logical, physical in enumerate(self.cache.block_table):
            count = min(pool.block_size, self.length - logical * pool.block_size)
            if count <= 0:
                break
            keys.append(pool.k[self.index, physical, :, :count])
            values.append(pool.v[self.index, physical, :, :count])
        return torch.cat(keys, dim=1)[None], torch.cat(values, dim=1)[None]
