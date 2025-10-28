import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define constants
EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = 224
train_images_dir = '/kaggle/input/aptos2019-blindness-detection/train_images'

# Focal Loss Implementation
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=5)
        y_true = tf.cast(y_true, tf.float32)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow((1 - y_pred), gamma)
        focal_loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    return focal_loss_fixed

# Dataset preparation
def create_tf_dataset(dataframe, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    def load_and_preprocess(img_id, label):
        img_path = tf.strings.join([train_images_dir + '/', img_id, '.png'])
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_saturation(img, 0.9, 1.1)
        return img, label

    img_ids = dataframe['id_code'].values
    labels = dataframe['diagnosis'].values
    dataset = tf.data.Dataset.from_tensor_slices((img_ids, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Assuming train_data and val_data are preloaded DataFrames
train_dataset = create_tf_dataset(train_data, shuffle=True, augment=True)
val_dataset = create_tf_dataset(val_data, shuffle=False, augment=False)

# Calculate class weights (optional backup)
class_weights = compute_class_weight('balanced', classes=np.unique(train_data['diagnosis']), y=train_data['diagnosis'])
class_weight_dict = dict(zip(np.unique(train_data['diagnosis']), class_weights))

# Compile model with focal loss
improved_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
]

# Train
print("Starting model training with FOCAL LOSS...")
history = improved_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)
print("Training completed!")

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['learning_rate'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()

# Summary
final_train_acc = max(history.history['accuracy'])
final_val_acc = max(history.history['val_accuracy'])
print(f"\nFinal Results:")
print(f"Best Training Accuracy: {final_train_acc:.4f}")
print(f"Best Validation Accuracy: {final_val_acc:.4f}")

# Save model with custom and standard loss
model_save_path = '/kaggle/working/diabetic_retinopathy_model.h5'
improved_model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

improved_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

keras_model_path = '/kaggle/working/diabetic_retinopathy_model.keras'
improved_model.save(keras_model_path)
print(f"Keras model saved to: {keras_model_path}")
