# SDNext Upscalers by chaiNNer

- Based on amazing [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) project  
- Original license at [LICENSE](https://github.com/chaiNNer-org/chaiNNer/blob/main/LICENSE)

## Install

Add to your extensions: <https://github.com/vladmandic/sd-extension-chainner>

## CLI Test

> ./chainner-test.py test/cutie.png

```log
model="SPSRNet-4x" arch="SPSRNet" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-SPSRNet-4x.png" (3140, 3144) time=5.14s
model="SwiftSR-2x" arch="Generator" scale=2
input="test/cutie.png" (785, 786) output="test/cutie-SwiftSR-2x.png" (1570, 1572) time=0.33s
model="DAT-4x" arch="DAT" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-DAT-4x.png" (3140, 3144) time=9.02s
model="HAT-2x" arch="HAT" scale=2
input="test/cutie.png" (785, 786) output="test/cutie-HAT-2x.png" (1570, 1572) time=6.13s
model="RealHAT-Sharper-4x" arch="HAT" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-RealHAT-Sharper-4x.png" (3140, 3144) time=6.10s
model="RRDBNet-4x" arch="RRDBNet" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-RRDBNet-4x.png" (3140, 3144) time=1.36s
model="SRFormer-Light-4x" arch="SRFormer" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-SRFormer-Light-4x.png" (3140, 3144) time=2.31s
model="RealHAT-GAN-4x" arch="HAT" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-RealHAT-GAN-4x.png" (3140, 3144) time=6.68s
model="HAT-L-3x" arch="HAT" scale=3
input="test/cutie.png" (785, 786) output="test/cutie-HAT-L-3x.png" (2355, 2358) time=12.18s
model="SRFormer-Nomos-4x" arch="SRFormer" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-SRFormer-Nomos-4x.png" (3140, 3144) time=6.24s
model="SwiftSR-4x" arch="Generator" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-SwiftSR-4x.png" (3140, 3144) time=0.50s
model="HAT-3x" arch="HAT" scale=3
input="test/cutie.png" (785, 786) output="test/cutie-HAT-3x.png" (2355, 2358) time=6.23s
model="HAT-L-4x" arch="HAT" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-HAT-L-4x.png" (3140, 3144) time=11.70s
model="HAT-L-2x" arch="HAT" scale=2
input="test/cutie.png" (785, 786) output="test/cutie-HAT-L-2x.png" (1570, 1572) time=12.17s
model="HAT-4x" arch="HAT" scale=4
input="test/cutie.png" (785, 786) output="test/cutie-HAT-4x.png" (3140, 3144) time=6.46s
```
