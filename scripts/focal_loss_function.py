import tensorflow as tf


def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-8
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Convert sparse labels to one-hot
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=5)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        weight = alpha * tf.pow((1 - y_pred), gamma)
        
        # Apply focal loss
        focal_loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fixed
