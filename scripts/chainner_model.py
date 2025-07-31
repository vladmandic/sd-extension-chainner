import os
from typing import TYPE_CHECKING
import PIL.Image
import numpy as np
from nodes.impl.upscale.tiler import MaxTileSize, NoTiling, Tiler
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl import image_utils
from nodes.load_model import load_model
from modules import devices, scripts, script_callbacks # pylint: disable=wrong-import-order
from modules.shared import opts, log, paths, readfile, OptionInfo # pylint: disable=wrong-import-order
from modules.upscaler import Upscaler, UpscalerData # pylint: disable=wrong-import-order
if TYPE_CHECKING:
    from nodes.impl.pytorch.types import PyTorchSRModel


predefined_models = os.path.join(scripts.basedir(), 'models.json')


def friendly_fullname(file: str):
    from urllib.parse import urlparse
    if "http" in file:
        file = urlparse(file).path
    file = os.path.basename(file)
    return file


class UpscalerChaiNNer(Upscaler):
    def __init__(self, dirname): # pylint: disable=unused-argument
        self.name = "chaiNNer"
        self.user_path = opts.data.get('chainner_models_path', os.path.join(paths.models_path, 'chaiNNer'))
        super().__init__()
        self.models = {}
        self.predefined = []
        self.fp16 = False
        self.scalers = self.find_scalers()

    def find_scalers(self):
        loaded = []
        scalers = []
        if self.predefined is None or len(self.predefined) == 0:
            self.predefined = readfile(predefined_models, silent=False)
        for model in self.predefined:
            local_name = os.path.join(self.user_path, friendly_fullname(model[1]))
            model_path = local_name if os.path.exists(local_name) else model[1]
            scaler = UpscalerData(name=f'{self.name} {model[0]}', path=model_path, upscaler=self)
            scalers.append(scaler)
            loaded.append(model_path)
        predefined = len(scalers)
        if not os.path.exists(self.user_path):
            return scalers
        downloaded = 0
        for fn in os.listdir(self.user_path): # from folder
            if not fn.endswith('.pth'):
                continue
            downloaded += 1
            file_name = os.path.join(self.user_path, fn)
            if file_name not in loaded:
                model_name = os.path.splitext(fn)[0]
                scaler = UpscalerData(name=f'{self.name} {model_name}', path=file_name, upscaler=self)
                scaler.custom = True
                scalers.append(scaler)
                loaded.append(file_name)
        discovered = len(scalers) - predefined
        log.debug(f'Available chaiNNer: path="{self.user_path}" defined={predefined} discovered={discovered} downloaded={downloaded}')
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

    def do_upscale(self, img: PIL.Image.Image, selected_model):
        devices.torch_gc()
        model = self.load_model(selected_model)
        if model is None:
            return img
        tile_size = opts.data.get('upscaler_tile_size', 192)
        try:
            with devices.inference_context(), devices.without_autocast():
                img_upscaled = pytorch_auto_split(img=np.array(img), model=model, device=devices.device, use_fp16=self.fp16, tiler=self.parse_tile_size_input(tile_size))
                if img_upscaled is None:
                    return img
                if np.isnan(img_upscaled).any():
                    log.error(f"Upscaler error: type={self.name} model={selected_model} device={devices.device} tile={tile_size} error=NaN")
                    return img
                img_norm = image_utils.to_uint8(img_upscaled, normalized=False)
                img = PIL.Image.fromarray(img_norm)
        except Exception as e:
            log.error(f"Upscaler error: type={self.name} model={selected_model} error={e}")
            from modules import errors
            errors.display(e, 'ChaiNNer')
        devices.torch_gc()
        if opts.data.get('upscaler_unload', False) and selected_model in self.models:
            del self.models[selected_model]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_model}")
            devices.torch_gc(force=True, reason='upscale')
        return img


def on_ui_settings():
    opts.add_option("chainner_models_path", OptionInfo(os.path.join(paths.models_path, 'chaiNNer'), "Folder with chaiNNer models", folder=True, section=('system-paths', "System Paths")))


script_callbacks.on_ui_settings(on_ui_settings)
