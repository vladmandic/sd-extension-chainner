from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import numpy as np
from ..upscale.auto_split import Tiler, auto_split
from .utils import np2tensor, tensor2np
if TYPE_CHECKING:
    from nodes.impl.pytorch.types import PyTorchModel


def pytorch_auto_split(img: np.ndarray | torch.Tensor, model: PyTorchModel, device: torch.device, use_fp16: bool, tiler: Tiler) -> np.ndarray | torch.Tensor:
    model = model.to(device)
    if use_fp16:
        model = model.half()

    def upscale(img: np.ndarray | torch.Tensor, _):
        use_tensor = False
        if isinstance(img, torch.Tensor):
            tensor = img
            use_tensor = True
        else:
            tensor = np2tensor(img, change_range=True)
        if use_fp16:
            tensor = tensor.half()
        tensor = tensor.to(device)
        result = model(tensor)
        if use_tensor:
            result = result.detach().cpu()
        else:
            result = tensor2np(result.detach().cpu().detach(), change_range=False, imtype=np.float32)
        del tensor
        return result

    return auto_split(img, upscale, tiler)
