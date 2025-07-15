<h1 align="center">Diabetic Retinopathy Detection Using Deep Learning</h1>

## Overview

Diabetic Retinopathy (DR) is a leading cause of vision impairment worldwide, affecting millions of people with diabetes. Early detection is critical for preventing vision loss, but manual diagnosis requires specialized expertise and is time-consuming.

This project presents an advanced deep learning solution leveraging **EfficientNet** and convolutional neural network, to classify retinal fundus images into five severity levels of DR. The model achieves strong performance through fine-tuning, focal loss for class imbalance, and comprehensive data augmentation, making it well-suited for real-world clinical applications.

<p align="center">
  <img src="images/main-picture.png"/>
</p>
---

## Key Features

- **Five-class classification:** No DR, Mild, Moderate, Severe, and Proliferative DR
- **EfficientNetB3 backbone:** Pre-trained on ImageNet and fine-tuned for optimal accuracy
- **Advanced data augmentation:** Enhances model robustness against varied imaging conditions
- **Focal loss function:** Mitigates class imbalance, improving detection of rare but critical cases
- **TensorFlow Dataset pipeline:** Efficient and scalable data loading with on-the-fly preprocessing
- **Grad-CAM visualization:** Interpretability through heatmaps highlighting image regions influencing predictions

---

## Dataset

The model is trained and validated on the publicly available [APTOS 2019 Blindness Detection Dataset], containing high-resolution retinal images labeled with DR severity levels.

---

## Methodology

1. **Data Exploration:** Analyzed image resolution and class distribution to inform preprocessing.
2. **Preprocessing:** Resized images to 224x224 pixels, normalized pixel values, and applied data augmentation techniques.
3. **Model Architecture:** Built upon EfficientNetB3 with top layers unfrozen for fine-tuning.
4. **Loss Function:** Implemented focal loss to focus learning on challenging and underrepresented classes.
5. **Training:** Used early stopping and learning rate scheduling to prevent overfitting and accelerate convergence.
6. **Evaluation:** Detailed metrics including accuracy, balanced accuracy, Cohen’s Kappa, and class-wise performance.
7. **Interpretability:** Generated Grad-CAM heatmaps to visualize model attention on retinal images.

---

## Results

- **Best Validation Accuracy:** ~74%
- **Balanced Accuracy:** ~50%, highlighting effective learning on imbalanced classes
- **Cohen’s Kappa:** ~0.59, demonstrating substantial agreement beyond chance
- **Per-class performance:** High accuracy on No DR and Moderate classes; ongoing work to improve rare class detection

---

## Installation & Usage

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd diabetic-retinopathy-detection
   ```

2. Install dependencies (recommended in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare dataset by placing retinal images in the `/train_images` directory.

4. Run training script:

   ```bash
   python train_model.py
   ```

5. For inference and Grad-CAM visualization, use:

   ```bash
   python predict_and_visualize.py
   ```

---

## File Structure

```
├── data/                   # Dataset directory (images, labels)
├── models/                 # Saved models (.h5, .keras)
├── notebooks/              # Jupyter notebooks for exploration & experimentation
├── scripts/
│   ├── train_model.py      # Training and evaluation pipeline
│   ├── predict_and_visualize.py  # Prediction and Grad-CAM visualization
│   └── focal_loss_function.py    # Custom focal loss code
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Future Work

- Improve recall on severe and proliferative DR through enhanced augmentation and balanced sampling
- Experiment with transformer-based architectures and ensemble methods
- Extend interpretability tools to generate textual explanations alongside heatmaps
- Deploy as a web app or mobile solution to aid clinical decision-making globally

---

## Acknowledgments

- The [APTOS 2019 dataset] for providing high-quality labeled images
- TensorFlow and Keras for their comprehensive deep learning frameworks
- The open-source community for invaluable tools and resources

---

*Empowering early diabetic retinopathy detection through cutting-edge AI — one pixel at a time.*
