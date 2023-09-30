#!/usr/bin/env python

import os
import sys
import time
import warnings
import torch
import numpy as np
from PIL import Image
from nodes.impl.upscale.tiler import MaxTileSize, NoTiling, Tiler
from nodes.impl.pytorch.auto_split import pytorch_auto_split
from nodes.impl.pytorch.types import PyTorchSRModel
from nodes.load_model import load_model
from nodes.log import logger


warnings.filterwarnings('ignore', category=UserWarning) # disable those for now as many backends reports tons
fp16 = False # HAT does not support fp16
device = torch.device('cuda')


def parse_tile_size_input(tile_size: int) -> Tiler:
    if tile_size == 0:
        return MaxTileSize(tile_size)
    elif tile_size == -1:
        return NoTiling()
    elif tile_size == -2:
        return MaxTileSize()
    elif tile_size < 0:
        raise ValueError(f"ChaiNNer invalid tile size: {tile_size}")
    return MaxTileSize(tile_size)


def upscale(image: Image, model: PyTorchSRModel, tile: int = 256):
    img = np.array(image)
    with torch.no_grad():
        upscaled = pytorch_auto_split(img, model=model, device=device, use_fp16=fp16, tiler=parse_tile_size_input(tile))
        return Image.fromarray(np.uint8(256 * upscaled))


if __name__ == "__main__":
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        logger.error('chainner: no files specified')
        sys.exit(1)
    for modelfile in os.listdir('models'):
        try:
            modelname = os.path.splitext(modelfile)[0]
            srmodel: PyTorchSRModel = load_model(os.path.join("models", modelfile), device=device, fp16=fp16)
            logger.info(f'model="{modelname}" arch="{srmodel.__class__.__name__}" scale={srmodel.scale}')
            for imagename in sys.argv:
                if not os.path.isfile(imagename):
                    logger.error(f'image={imagename} not found')
                    continue
                inputimage = Image.open(imagename).convert('RGB')
                t0 = time.time()
                outputimage = upscale(image=inputimage, model=srmodel, tile=256)
                t1 = time.time()
                base, ext = os.path.splitext(imagename)
                outputname = f'{base}-{modelname}{ext}'
                outputimage.save(outputname)
                logger.info(f'input="{imagename}" {inputimage.size} output="{outputname}" {outputimage.size} time={t1-t0:.2f}s')
            srmodel = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # sys.exit(1)
        except Exception as e:
            logger.error(f'Error: fn={modelfile} {e}')
