import os
import PIL.Image
import numpy as np
import torch
from modules import devices, modelloader, script_callbacks
from modules.shared import opts, log, OptionInfo, paths
from modules.upscaler import Upscaler, UpscalerData
from nodes.impl.upscale.tiler import MaxTileSize, NoTiling, Tiler
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.load_model import load_model
from nodes.impl.pytorch.types import PyTorchSRModel


chainner_models = [
    ["4x DAT", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/DAT-4x.pth"],
    ["2x HAT", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-2x.pth"],
    ["3x HAT", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-3x.pth"],
    ["4x HAT", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-4x.pth"],
    ["2x HAT-Large", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-2x.pth"],
    ["3x HAT-Large", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-3x.pth"],
    ["4x HAT-Large", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/HAT-L-4x.pth"],
    ["4x RRDBNet", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RRDBNet-4x.pth"],
    ["4x RealHAT-GAN", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealHAT-GAN-4x.pth"],
    ["4x RealHAT-Sharper", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealHAT-Sharper-4x.pth"],
    ["4x SPSRNet", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SPSRNet-4x.pth"],
    ["4x SRFormer-Light", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SRFormer-Light-4x.pth"],
    ["4x SRFormer-Nomos", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SRFormer-Nomos-4x.pth"],
    ["2x SwiftSR", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SwiftSR-2x.pth"],
    ["4x SwiftSR", "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/SwiftSR-4x.pth"],
]


class UpscalerChaiNNer(Upscaler):
    def __init__(self, dirname): # pylint: disable=unused-argument
        self.name = "chaiNNer"
        self.user_path = opts.data.get('chainner_models_path', os.path.join(paths.models_path, 'chaiNNer'))
        super().__init__()
        self.models = {}
        self.fp16 = False
        self.scalers = self.find_scalers()

    def find_scalers(self):
        loaded = []
        scalers = []
        for model in chainner_models:
            local_name = os.path.join(self.user_path, modelloader.friendly_fullname(model[1]))
            model_path = local_name if os.path.exists(local_name) else model[1]
            scaler = UpscalerData(name=f'{self.name} {model[0]}', path=model_path, upscaler=self)
            scalers.append(scaler)
            loaded.append(model_path)
        for fn in os.listdir(self.user_path): # from folder
            if not fn.endswith('.pth'):
                continue
            file_name = os.path.join(self.user_path, fn)
            if file_name not in loaded:
                model_name = os.path.splitext(fn)[0]
                scaler = UpscalerData(name=f'{self.name} {model_name}', path=file_name, upscaler=self)
                scaler.custom = True
                scalers.append(scaler)
                loaded.append(file_name)
        return scalers

    def load_model(self, path: str):
        info = self.find_model(path)
        if info is None:
            return
        if self.models.get(info.local_data_path, None) is not None:
            log.debug(f"Upscaler cached: type={self.name} model={info.local_data_path}")
            model=self.models[info.local_data_path]
        else:
            model: PyTorchSRModel = load_model(info.local_data_path, device=devices.device, fp16=self.fp16)
            log.info(f"Upscaler loaded: type={self.name} model='{info.local_data_path}'")
            self.models[info.local_data_path] = model
        return model

    def parse_tile_size_input(self, tile_size: int) -> Tiler:
        if tile_size == 0:
            return MaxTileSize(tile_size)
        elif tile_size == -1:
            return NoTiling()
        elif tile_size == -2:
            return MaxTileSize()
        elif tile_size < 0:
            raise ValueError(f"ChaiNNer invalid tile size: {tile_size}")
        return MaxTileSize(tile_size)

    def do_upscale(self, img: PIL.Image.Image, selected_file):
        devices.torch_gc()
        model = self.load_model(selected_file)
        if model is None:
            return img
        np_img = np.array(img)
        tile_size = opts.data.get('upscaler_tile_size', 192)
        with torch.no_grad():
            upscaled = pytorch_auto_split(img=np_img, model=model, device=devices.device, use_fp16=self.fp16, tiler=self.parse_tile_size_input(tile_size))
            img = PIL.Image.fromarray(np.uint8(256 * upscaled))
        devices.torch_gc()
        if opts.data.get('upscaler_unload', False) and selected_file in self.models:
            del self.models[selected_file]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_file}")
            devices.torch_gc(force=True)
        return img


def on_ui_settings():
    opts.add_option("chainner_models_path", OptionInfo(os.path.join(paths.models_path, 'chaiNNer'), "Folder with chaiNNer models", folder=True, section=('system-paths', "System Paths")))


script_callbacks.on_ui_settings(on_ui_settings)
