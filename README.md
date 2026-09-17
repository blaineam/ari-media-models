# Ari media models

Core ML models that [Ari Helper](https://blainem.com/apps/ari-helper/) downloads the first time a Photo Studio tool needs one. The networks and their weights are the upstream authors' work, under the upstream licence.

| File | What it does | Input and output | Upstream weights | Licence |
|---|---|---|---|---|
| `NAFNet-Denoise.mlpackage.zip` | Removes sensor noise | 512×512 RGB image | NAFNet-SIDD-width32 | MIT |
| `NAFNet-Deblur.mlpackage.zip` | Sharpens motion blur | 512×512 RGB image | NAFNet-GoPro-width32 | MIT |

The files are attached to the [releases](../../releases), not committed.

## How they were made

`convert.py` loads the official checkpoints from [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) into `nafnet_arch_clean.py`. That file is the upstream architecture with a plain LayerNorm, so the model can be traced, and without the training-only local-statistics variant. The script traces the network at a fixed 512-pixel tile and converts it with coremltools 9 at **float32**. Earlier half-precision conversions overflowed and returned noise.

It then checks the result against the PyTorch output. On a portrait with added noise (σ = 25), PSNR goes from 20.6 dB to 29.5 dB. On an 8-pixel motion blur, it goes from 26.6 dB to 28.9 dB. The CPU, GPU and Neural Engine give the same results.

## Credits

NAFNet: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang and Jian Sun, *Simple Baselines for Image Restoration*, ECCV 2022. © 2022 megvii-model, MIT licence ([LICENSE-NAFNet](LICENSE-NAFNet)).
