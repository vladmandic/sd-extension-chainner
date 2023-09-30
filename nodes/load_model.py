import os
from typing import Tuple
import torch
from nodes.impl.pytorch.model_loading import load_state_dict
from nodes.impl.pytorch.types import PyTorchModel
from nodes.utils.unpickler import RestrictedUnpickle


def parse_ckpt_state_dict(checkpoint: dict):
    state_dict = {}
    for i, j in checkpoint.items():
        if "netG." in i:
            key = i.replace("netG.", "")
            state_dict[key] = j
        elif "module." in i:
            key = i.replace("module.", "")
            state_dict[key] = j
    return state_dict


def load_model(path: str, device, fp16: bool = False) -> Tuple[PyTorchModel, str, str]:
    """Read a pth file from the specified path and return it as a state dict
    and loaded model after finding arch config"""
    assert os.path.exists(path), f"Model file at location {path} does not exist"
    assert os.path.isfile(path), f"Path {path} is not a file"
    try:
        extension = os.path.splitext(path)[1].lower()

        if extension == ".pt":
            state_dict = torch.jit.load(  # type: ignore
                path, map_location=device
            ).state_dict()
        elif extension == ".pth":
            state_dict = torch.load(
                path,
                map_location=device,
                pickle_module=RestrictedUnpickle,  # type: ignore
            )
        elif extension == ".ckpt":
            checkpoint = torch.load(
                path,
                map_location=device,
                pickle_module=RestrictedUnpickle,  # type: ignore
            )
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            state_dict = parse_ckpt_state_dict(checkpoint)
        else:
            raise ValueError(
                f"Unsupported model file extension {extension}. Please try a supported model type."
            )

        model = load_state_dict(state_dict)

        for _, v in model.named_parameters():
            v.requires_grad = False
        model.eval()
        model = model.to(device)
        should_use_fp16 = fp16
        if should_use_fp16:
            model = model.half()
        else:
            model = model.float()
    except Exception as e:
        raise ValueError(
            f"Model {os.path.basename(path)} is unsupported by chaiNNer. Please try"
            " another."
        ) from e

    return model
