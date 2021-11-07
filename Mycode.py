import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.optimizers import Adam

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

def build_model():
    filters = 56
    units = 24
    kernel_size = 5
    learning_rate = 1e-2
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    return model

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE = 128

train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
eval_dataset = mnist_test.map(scale).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = build_model()

epochs=5
model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=num_train_examples/epochs,
        validation_steps=num_test_examples/epochs,
        epochs=epochs)
