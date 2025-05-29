import numpy as np
import tensorflow as tf

# Parameters
n_samples = 1000
batch_size = 100
num_steps = 1000

# Generate sample data
X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1)).astype(np.float32)

# Define the model
class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.k = tf.Variable(tf.random.normal([1, 1]), name='slope')
        self.b = tf.Variable(tf.zeros([1, 1]), name='bias')

    def call(self, X):
        return tf.matmul(X, self.k) + self.b

model = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
for step in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        loss_val = loss_fn(y_batch, y_pred)

    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if (step + 1) % 100 == 0:
        print(f'Step {step+1}: Loss = {loss_val:.4f}, k = {model.k.numpy()[0,0]:.4f}, b = {model.b.numpy()[0,0]:.4f}')

print(f'Final model: y = {model.k.numpy()[0,0]:.4f}x + {model.b.numpy()[0,0]:.4f}')