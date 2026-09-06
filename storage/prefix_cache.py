"""Bounded immutable prefix snapshots with shared physical blocks."""
from collections import OrderedDict


class PrefixStore:
    def __init__(self, max_entries=16):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = 0
        self.reused_tokens = 0

    def put(self, tokens, cache):
        key = tuple(tokens)
        if not key or not self.max_entries or key in self.entries:
            return
        self.entries[key] = cache.fork(len(key))
        while len(self.entries) > self.max_entries:
            _, old = self.entries.popitem(last=False)
            old.close()

    def get(self, tokens):
        # Keep at least one prompt token unprocessed to reconstruct next logits.
        matches = [key for key in self.entries if len(key) < len(tokens) and tuple(tokens[:len(key)]) == key]
        if not matches:
            return None
        key = max(matches, key=len)
        self.entries.move_to_end(key)
        self.hits += 1
        self.reused_tokens += len(key)
        return self.entries[key].fork()

    @property
    def pinned_blocks(self):
        return len({b for cache in self.entries.values() for b in cache.block_table})

    def clear(self):
        for cache in self.entries.values():
            cache.close()
        self.entries.clear()
