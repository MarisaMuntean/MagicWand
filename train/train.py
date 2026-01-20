import tensorflow as tf
import pandas as pd
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# Set dataset shape
timesteps = 200
lr = 1e-4
num_epochs = 200
batch_size = 4
input_dim = 2  # Number of sensor data dimensions (e.g., if only X and Z axes are recorded, set to 2)
num_classes = 5  # Number of classes (e.g., if recording 4 actions, set to 5, etc.)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,3), strides=(1,1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(16, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.expand_dims(x, axis=2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Load training and test data (replace filenames as needed)
    train_x_pd = pd.read_csv('data_x.csv', header=None)  # Training input data file
    train_y_pd = pd.read_csv('data_y.csv', header=None)  # Training label file
    test_x_pd = pd.read_csv('test_x.csv', header=None)   # Test input data file (can reuse training data if needed)
    test_y_pd = pd.read_csv('test_y.csv', header=None)   # Test label file

    # Convert and reshape data
    train_x = tf.convert_to_tensor(train_x_pd.to_numpy(), dtype=tf.float32)
    train_x = tf.reshape(train_x, [-1, timesteps, input_dim])
    train_y = tf.convert_to_tensor(train_y_pd.to_numpy(), dtype=tf.int32)

    test_x = tf.convert_to_tensor(test_x_pd.to_numpy(), dtype=tf.float32)
    test_x = tf.reshape(test_x, [-1, timesteps, input_dim])
    test_y = tf.convert_to_tensor(test_y_pd.to_numpy(), dtype=tf.int32)

    # Create datasets
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_data = test_data.batch(batch_size)

    # Calculate dataset sizes and steps
    train_size = tf.data.experimental.cardinality(train_data).numpy()
    test_size = tf.data.experimental.cardinality(test_data).numpy()
    steps_per_epoch = math.ceil(train_size / batch_size)
    validation_steps = math.ceil(test_size / batch_size)

    train_data = train_data.repeat()
    test_data = test_data.repeat()

    # Build model
    model = Model()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Train model
    model.fit(
        train_data,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_data,
        validation_steps=validation_steps
    )

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)