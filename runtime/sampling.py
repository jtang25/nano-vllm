from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")

    def probabilities(self, logits):
        if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
            raise ValueError("expected finite [V] logits")
        if self.temperature == 0:
            result = torch.zeros_like(logits, dtype=torch.float32)
            result[logits.argmax()] = 1
            return result
        scores = logits.float()
        scores = (scores - scores.max()) / self.temperature
        if self.top_k is not None:
            if self.top_k > logits.numel():
                raise ValueError("top_k exceeds vocabulary")
            values, indices = scores.topk(self.top_k)
            scores = torch.full_like(scores, -torch.inf).scatter(0, indices, values)
        if self.top_p < 1:
            sorted_scores, order = scores.sort(descending=True)
            cumulative = sorted_scores.softmax(-1).cumsum(-1)
            remove = cumulative - sorted_scores.softmax(-1) >= self.top_p
            sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
            scores = torch.empty_like(scores).scatter(0, order, sorted_scores)
        return scores.softmax(-1)


def draw(probabilities, generator=None):
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def residual_distribution(target, draft):
    residual = (target - draft).clamp_min(0)
    mass = residual.sum()
    if mass <= 0:
        raise ValueError("no rejection mass: identical distributions cannot reject")
    return residual / mass
