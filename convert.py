import sys, torch, numpy as np, coremltools as ct
from PIL import Image
from nafnet_arch_clean import NAFNet
sp = sys.argv[1]
configs = {
  'sidd32': dict(width=32, enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2]),
  'gopro32': dict(width=32, enc_blk_nums=[1,1,1,28], middle_blk_num=1, dec_blk_nums=[1,1,1,1]),
}
class Wrapped(torch.nn.Module):
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, x):
        n = self.net
        inp = x / 255.0
        h = n.intro(inp)
        skips = []
        for encoder, down in zip(n.encoders, n.downs):
            h = encoder(h); skips.append(h); h = down(h)
        h = n.middle_blks(h)
        for decoder, up, skip in zip(n.decoders, n.ups, skips[::-1]):
            h = up(h); h = h + skip; h = decoder(h)
        h = n.ending(h) + inp
        return h.clamp(0, 1) * 255.0
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)).permute(2,0,1)[None]
def psnr(a, b): return float(10 * torch.log10(255**2 / ((a - b) ** 2).mean()))
clean = load(f'{sp}/face-original.png')
inputs = {'sidd32': load(f'{sp}/mprnet/in-noisy.png'), 'gopro32': load(f'{sp}/mprnet/in-blurred.png')}
names = {'sidd32': 'NAFNet-Denoise', 'gopro32': 'NAFNet-Deblur'}
for key, cfg in configs.items():
    net = NAFNet(**cfg)
    state = torch.load(f'{key}.pth', map_location='cpu', weights_only=False)
    state = state.get('params', state)
    missing = net.load_state_dict(state, strict=True)
    model = Wrapped(net).eval()
    with torch.no_grad():
        out = model(inputs[key])
    print(key, 'torch psnr in', round(psnr(inputs[key], clean), 2), 'out', round(psnr(out, clean), 2), flush=True)
    traced = torch.jit.trace(model, torch.rand(1, 3, 512, 512) * 255)
    ml = ct.convert(traced,
        inputs=[ct.ImageType(name='image', shape=(1, 3, 512, 512), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name='restored', color_layout=ct.colorlayout.RGB)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17)
    ml.author = 'megvii-research (NAFNet, MIT); converted for Ari Helper'
    ml.license = 'MIT'
    ml.short_description = f'{names[key]}: NAFNet width32 ({key}), 512x512 RGB in and out, float32.'
    ml.save(f'{names[key]}.mlpackage')
    pred = ml.predict({'image': Image.open(f'{sp}/mprnet/' + ('in-noisy.png' if key == 'sidd32' else 'in-blurred.png')).convert('RGB')})['restored']
    arr = torch.from_numpy(np.asarray(pred.convert('RGB'), dtype=np.float32)).permute(2,0,1)[None]
    print(key, 'coreml psnr out', round(psnr(arr, clean), 2), flush=True)
    pred.save(f'out-{names[key]}.png')
