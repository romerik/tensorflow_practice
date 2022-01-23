#coding:utf-8

import tensorflow as tf
import keras_tuner as kt

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

print(training_images.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

print("\n\n")

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications.shape)
print(classifications[0])
print(test_labels[0])

"""
def build_model(hp):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
  model.add(tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=512, step=32)))
  model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

tuner = kt.RandomSearch(build_model, objective='loss', max_trials=5)

tuner.search(training_images, training_labels, epochs=5)
tuner.get_best_models(num_models=3)

"""
