import torch


def get_gpu_info():
    """Print and return GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {"gpu_name": gpu_name, "total_memory_gb": round(total_mem, 2), "available": True}
    return {"available": False}


def log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory — Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
