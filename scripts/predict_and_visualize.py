import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Class labels for diabetic retinopathy levels
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Visualize a few model predictions with confidence scores
def visualize_predictions(improved_model, val_dataset, n_samples=8):
    plt.figure(figsize=(20, 10))
    
    sample_batch = next(iter(val_dataset.take(1)))
    images, true_labels = sample_batch
    predictions = improved_model.predict(images[:n_samples], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    for i in range(n_samples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        confidence = np.max(predictions[i]) * 100
        true_class = class_names[int(true_labels[i])]
        pred_class = class_names[int(pred_classes[i])]
        color = 'green' if true_labels[i] == pred_classes[i] else 'red'
        
        plt.title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%', 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Ultra-simple Grad-CAM approximation using input gradients
def ultra_simple_gradcam(model, img_array, class_index):
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, img_array)
    grads = tf.abs(grads)
    grads = tf.reduce_mean(grads, axis=-1)[0]
    grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads))
    return grads.numpy()

# Display Grad-CAM result side by side
def display_simple_results(img, heatmap, true_class, pred_class, confidence):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Original\nTrue: {class_names[true_class]}', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Attention Map', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.title(f'Predicted: {class_names[pred_class]}\n{confidence:.1%}', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load model
    print("Loading trained model...")
    model = tf.keras.models.load_model('models/diabetic_retinopathy_model.keras')
    model.build(input_shape=(None, 224, 224, 3))
    print("✓ Model loaded and built.")

    # Load sample validation data
    print("Loading sample validation data...")
    sample_images = np.load('models/sample_images.npy')
    sample_labels = np.load('models/sample_labels.npy')
    
    # Convert to tf.data.Dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((sample_images, sample_labels))
    val_dataset = val_dataset.batch(8)
    
    print("\nRunning prediction visualization...")
    visualize_predictions(model, val_dataset)

    print("\nRunning Grad-CAM visualization...")
    val_batch = next(iter(val_dataset))
    images, labels = val_batch
    predictions = model.predict(images[:3], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    
    for i in range(3):
        print(f"\nSample {i+1}:")
        print(f"True: {class_names[int(labels[i])]}")
        print(f"Predicted: {class_names[int(pred_classes[i])]}")
        print(f"Confidence: {np.max(predictions[i]):.1%}")
        
        img_array = tf.convert_to_tensor(np.expand_dims(images[i], axis=0))
        heatmap = ultra_simple_gradcam(model, img_array, pred_classes[i])
        
        display_simple_results(
            images[i].numpy(),
            heatmap,
            int(labels[i]),
            pred_classes[i],
            np.max(predictions[i])
        )

if __name__ == '__main__':
    main()
