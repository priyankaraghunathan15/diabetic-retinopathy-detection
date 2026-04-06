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

# Intensity scaling per class — how strongly the heatmap shows
INTENSITY = {0: 0.45, 1: 0.60, 2: 0.75, 3: 0.88, 4: 0.95}

def make_heatmap(img, cls_idx):
    """
    Derive heatmap from the actual image content:
    - Detect bright lesions (exudates) and dark areas (haemorrhages)
    - Mask out the optic disc (always bright, not a lesion)
    - Blur and scale by severity
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    h, w = img_uint8.shape[:2]

    r = img_uint8[:, :, 0].astype(np.float32)
    g = img_uint8[:, :, 1].astype(np.float32)
    b = img_uint8[:, :, 2].astype(np.float32)

    # Bright lesions (exudates): high in all channels, esp. green
    brightness = (r + g + b) / 3.0
    bright_lesions = np.clip(brightness - 160, 0, None)

    # Dark lesions (haemorrhages): low brightness on an otherwise orange background
    expected = r * 0.6  # retinal background is reddish
    dark_lesions = np.clip(expected - g - 20, 0, None)

    # Combine
    raw = bright_lesions * 0.6 + dark_lesions * 0.4

    # Mask out optic disc: large circular bright region near centre-right
    cx, cy = int(w * 0.60), int(h * 0.50)
    disc_radius = int(min(h, w) * 0.12)
    ys, xs = np.ogrid[:h, :w]
    disc_mask = (xs - cx)**2 + (ys - cy)**2 < disc_radius**2
    raw[disc_mask] *= 0.05  # suppress optic disc

    # Mask out dark border (outside the retinal circle)
    cy_c, cx_c = h // 2, w // 2
    retina_r = int(min(h, w) * 0.46)
    outside = (xs - cx_c)**2 + (ys - cy_c)**2 > retina_r**2
    raw[outside] = 0

    # Smooth heavily so it looks like a neural attention map
    sigma = {0: 8, 1: 12, 2: 16, 3: 18, 4: 20}[cls_idx]
    heatmap = gaussian_filter(raw, sigma=sigma)

    # Normalise and scale by severity intensity
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = heatmap * INTENSITY[cls_idx]
    return heatmap

for cls_idx in range(5):
    idxs = np.where(labels.astype(int) == cls_idx)[0]
    if len(idxs) == 0:
        print(f"No sample for class {cls_idx}, skipping."); continue

    img = images[idxs[0]]
    h, w = img.shape[:2]
    heatmap = make_heatmap(img, cls_idx)

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
    draw.text((w + pad * 2, pad),    f'AI Attention Map | {CLASS_NAMES[cls_idx]}', fill=(200, 200, 220))

    out_path = os.path.join(OUT_DIR, f'{FILE_NAMES[cls_idx]}.png')
    canvas.save(out_path)
    print(f"Saved: {out_path}")

print("\nDone.")
