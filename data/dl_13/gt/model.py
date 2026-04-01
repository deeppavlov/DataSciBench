
import tensorflow as tf


class LinearRegressionModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(0.8, dtype=tf.float32)
        self.b = tf.Variable(0.7, dtype=tf.float32)

    def __call__(self, x):
        return self.w * x + self.b

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

@tf.function
def train_step(model, x, y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * gradients[0])
    model.b.assign_sub(learning_rate * gradients[1])
    return loss
