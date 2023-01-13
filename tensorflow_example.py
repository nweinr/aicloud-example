# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras

def build_model():
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    return model


def load_data():
    
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    return x_train, y_train, x_test, y_test


def train(model, x_train, y_train, x_test, y_test):
    
    model.fit(x_train, y_train, epochs=5)
    
    model.evaluate(x_test,  y_test, verbose=2)
    
    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    
    model.save('trained_model')


def predict(x):
    
    model = keras.models.load_model('trained_model')
    
    prob = model.predict(x)
    pred = np.argmax(prob, axis=1)
    
    return pred, prob


if __name__ == '__main__':
    
    # Check that we are using the GPU
    print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
    
    model = build_model()

    x_train, y_train, x_test, y_test = load_data()
    
    train(model, x_train, y_train, x_test, y_test)