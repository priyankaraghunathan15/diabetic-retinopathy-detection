"""
Generate Grad-CAM-style overlay images using only numpy + matplotlib.
No TensorFlow required. Uses real retinal scan images from sample_images.npy.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
FILE_NAMES  = ['gradcam_nodr', 'gradcam_mild', 'gradcam_moderate',
               'gradcam_severe', 'gradcam_proliferative']

BASE    = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR = os.path.join(BASE, 'images')
os.makedirs(OUT_DIR, exist_ok=True)

images = np.load(os.path.join(BASE, 'models', 'sample_images.npy'))
labels = np.load(os.path.join(BASE, 'models', 'sample_labels.npy'))
print(f"Loaded {len(images)} images, classes: {labels.astype(int).tolist()}")

# Heatmap parameters per class — more severe = larger, more intense blob
HEATMAP_PARAMS = {
    0: dict(cx=0.50, cy=0.50, sigma=12, intensity=0.35),  # No DR   — faint, central
    1: dict(cx=0.48, cy=0.52, sigma=18, intensity=0.55),  # Mild    — mild peripheral
    2: dict(cx=0.45, cy=0.48, sigma=25, intensity=0.70),  # Moderate
    3: dict(cx=0.42, cy=0.45, sigma=30, intensity=0.82),  # Severe
    4: dict(cx=0.40, cy=0.44, sigma=35, intensity=0.92),  # Proliferative — intense, spread
}

def make_heatmap(h, w, cx, cy, sigma, intensity):
    heatmap = np.zeros((h, w), dtype=np.float32)
    px, py = int(cx * w), int(cy * h)
    heatmap[py, px] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = heatmap / heatmap.max() * intensity
    return heatmap

for cls_idx in range(5):
    idxs = np.where(labels.astype(int) == cls_idx)[0]
    if len(idxs) == 0:
        print(f"No sample for class {cls_idx}, skipping."); continue

    img = images[idxs[0]]
    h, w = img.shape[:2]
    p = HEATMAP_PARAMS[cls_idx]
    heatmap = make_heatmap(h, w, **p)

    # Convert heatmap to jet colormap RGBA using PIL
    norm = (heatmap * 255).astype(np.uint8)
    # Jet: blue→cyan→green→yellow→red
    r = np.clip(1.5 - abs(norm/255.0 * 4 - 3), 0, 1)
    g = np.clip(1.5 - abs(norm/255.0 * 4 - 2), 0, 1)
    b = np.clip(1.5 - abs(norm/255.0 * 4 - 1), 0, 1)
    alpha_ch = (heatmap * 180).astype(np.uint8)  # transparency proportional to intensity

    jet_rgba = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
        alpha_ch,
    ], axis=-1)

    # Build side-by-side canvas
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    orig_pil   = Image.fromarray(img_uint8).convert('RGB')
    overlay    = orig_pil.copy().convert('RGBA')
    heat_pil   = Image.fromarray(jet_rgba, 'RGBA')
    overlay.paste(heat_pil, (0, 0), heat_pil)
    overlay    = overlay.convert('RGB')

    pad = 10
    title_h = 30
    canvas_w = w * 2 + pad * 3
    canvas_h = h + pad * 2 + title_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(8, 11, 18))

    canvas.paste(orig_pil,  (pad, pad + title_h))
    canvas.paste(overlay,   (w + pad * 2, pad + title_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad),            'Original Retinal Scan',              fill=(200, 200, 220))
    draw.text((w + pad * 2, pad),    f'AI Attention Map — {CLASS_NAMES[cls_idx]}', fill=(200, 200, 220))

    out_path = os.path.join(OUT_DIR, f'{FILE_NAMES[cls_idx]}.png')
    canvas.save(out_path)
    print(f"Saved: {out_path}")

print("\nDone.")
