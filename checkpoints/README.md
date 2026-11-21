# Checkpoints

One file per (method, dataset, fold). Each carries the learnable tensors outside the ConvNeXt-T backbone (concept thresholds, head A/W, pigment U-Net, residual projection, PGA stubs), plus the training config and the per-seed test metrics used by the Nadeau–Bengio corrected paired test.

To load a checkpoint at inference, attach the standard timm ConvNeXt-T ImageNet-1k backbone named in `cfg.backbone_source`:

```python
import pickle, timm, torch
from pigmentnet import PigmentNet
ck = pickle.loads(open('pigmentnet_D1_fold0.pt', 'rb').read())
model = PigmentNet()
backbone = timm.create_model('convnext_tiny.fb_in1k', pretrained=True)
# transfer the backbone weights into model.texture, then load ck['state_dict']
```

`sha256sums.txt` lists every file. The per-fold per-seed metrics are mirrored in `results/fold_level_metrics.json` so the statistical tests in `scripts/nadeau_bengio.py` can run without loading any state dict.
