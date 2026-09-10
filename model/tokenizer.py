class ByteTokenizer:
    """UTF-8 bytes give draft and target an identical, dependency-free vocabulary."""
    vocab_size = 259
    bos_id = 256
    eos_id = 257
    pad_id = 258
    name = "utf8-bytes-v1"

    def encode(self, text, bos=True, eos=False):
        return ([self.bos_id] if bos else []) + list(text.encode("utf-8")) + ([self.eos_id] if eos else [])

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, spec):
        import hashlib
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"], trust_remote_code=False)
        digest = hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        if digest != spec["backend_sha256"]:
            raise ValueError("tokenizer differs from the checkpoint's pinned tokenizer")
        self.eos_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def tokenizer_for(model):
    spec = getattr(model, "tokenizer_spec", ByteTokenizer.name)
    return HFTokenizer(spec) if isinstance(spec, dict) else ByteTokenizer()


def check_tokenizers(target, draft):
    a, b = getattr(target, "tokenizer_spec", None), getattr(draft, "tokenizer_spec", None)
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict) or a["backend_sha256"] != b["backend_sha256"]:
            raise ValueError("draft/target tokenizer mappings differ")
