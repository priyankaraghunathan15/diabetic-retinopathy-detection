"""
Generate one Grad-CAM image per DR severity class and save to images/.
Outputs: gradcam_nodr.png, gradcam_mild.png, gradcam_moderate.png,
         gradcam_severe.png, gradcam_proliferative.png
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
FILE_NAMES  = ['gradcam_nodr', 'gradcam_mild', 'gradcam_moderate',
               'gradcam_severe', 'gradcam_proliferative']
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')

# ── load model & data ──────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetic_retinopathy_model.keras')
)
model.build(input_shape=(None, 224, 224, 3))
print("Model loaded.")

print("Loading sample data...")
images = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_images.npy'))
labels = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_labels.npy'))
print(f"Loaded {len(images)} samples. Class distribution: {np.bincount(labels.astype(int))}")

# ── grad-cam (input-gradient method) ──────────────────────────────────
def gradcam(model, img_array, class_idx):
    inp = tf.convert_to_tensor(img_array[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        preds = model(inp)
        loss  = preds[:, class_idx]
    grads = tape.gradient(loss, inp)
    grads = tf.abs(grads)
    heatmap = tf.reduce_mean(grads, axis=-1)[0].numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# ── generate one image per class ──────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

for cls_idx in range(5):
    idxs = np.where(labels.astype(int) == cls_idx)[0]
    if len(idxs) == 0:
        print(f"WARNING: no samples found for class {cls_idx} ({CLASS_NAMES[cls_idx]}), skipping.")
        continue

    img = images[idxs[0]]
    preds = model.predict(img[np.newaxis], verbose=0)[0]
    pred_cls  = int(np.argmax(preds))
    confidence = float(np.max(preds))
    heatmap = gradcam(model, img, pred_cls)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#080b12')

    axes[0].imshow(img)
    axes[0].set_title('Original Retinal Scan', color='white', fontsize=12, pad=10)
    axes[0].axis('off')

    axes[1].imshow(img)
    axes[1].imshow(heatmap, cmap='jet', alpha=0.45)
    axes[1].set_title(
        f'AI Attention Map\n{CLASS_NAMES[pred_cls]} · {confidence:.0%} confidence',
        color='white', fontsize=12, pad=10
    )
    axes[1].axis('off')

    plt.tight_layout(pad=1.5)
    out_path = os.path.join(OUT_DIR, f'{FILE_NAMES[cls_idx]}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#080b12')
    plt.close()

    print(f"[{cls_idx}] {CLASS_NAMES[cls_idx]:15s} → pred: {CLASS_NAMES[pred_cls]:15s} "
          f"conf: {confidence:.1%}  saved: {out_path}")

print("\nDone. All Grad-CAM images saved to images/")
