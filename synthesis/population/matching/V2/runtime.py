"""Optional Torch dependency and shared batch conversion."""

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = nn = F = None


def to_tensors(x, indices, device):
    return {key: torch.tensor(value[indices], device=device,
                                 dtype=torch.float32 if key == "cont" else torch.long)
            for key, value in x.items() if key != "cardinalities"}
