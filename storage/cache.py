import torch


class KVCache:
    """Fixed-capacity K/V storage for one attention layer and one batch."""

    def __init__(self, batch_size, n_kv_heads, capacity, d_head, *, device, dtype):
        if min(batch_size, n_kv_heads, capacity, d_head) <= 0:
            raise ValueError("cache dimensions must be positive")

        shape = (batch_size, n_kv_heads, capacity, d_head)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.capacity = capacity
        self.length = 0

    def append(self, k_new, v_new):
        """Append [B, Hkv, T, Dh] and return populated K/V views."""
        if k_new.shape != v_new.shape or k_new.ndim != 4:
            raise ValueError("K and V must have matching [B, Hkv, T, Dh] shapes")
        if k_new.shape[:2] != self.k.shape[:2] or k_new.shape[-1] != self.k.shape[-1]:
            raise ValueError("K/V shape does not match the cache")
        if any(x.device != self.k.device or x.dtype != self.k.dtype for x in (k_new, v_new)):
            raise ValueError("K/V device and dtype must match the cache")

        start = self.length
        end = start + k_new.shape[2]
        if end > self.capacity:
            raise ValueError(f"cache capacity exceeded: need {end}, have {self.capacity}")

        self.k[:, :, start:end].copy_(k_new)
        self.v[:, :, start:end].copy_(v_new)
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self):
        self.length = 0

    def truncate(self, length):
        if not 0 <= length <= self.length:
            raise ValueError("invalid rollback length")
        self.length = length

    @property
    def allocated_bytes(self):
        return 2 * self.k.numel() * self.k.element_size()

    @property
    def live_bytes(self):
        return self.allocated_bytes // self.capacity * self.length
