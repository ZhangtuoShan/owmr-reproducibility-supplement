import torch


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_torch_device(device=None, gpu_id: int = -1) -> torch.device:
    """Resolve ElegantRL's legacy gpu_id plus optional cpu/cuda/mps selector."""
    if device is None:
        return torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    device = str(device).lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        if torch.cuda.is_available() and gpu_id >= 0:
            return torch.device(f"cuda:{gpu_id}")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps":
        if not _mps_is_available():
            raise RuntimeError("PyTorch MPS is not available in this environment.")
        return torch.device("mps")
    if device == "cuda":
        if not (torch.cuda.is_available() and gpu_id >= 0):
            raise RuntimeError("CUDA is not available, or gpu_id is negative.")
        return torch.device(f"cuda:{gpu_id}")
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this environment.")
        return torch.device(device)

    raise ValueError(f"Unsupported device selector: {device}")
