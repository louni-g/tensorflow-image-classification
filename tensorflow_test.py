# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.cifar10
cifar10_data = data.load_data()

(train_images, train_labels), (test_images, test_labels) = cifar10_data

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog","horse", "ship", "truck"]


"""
index = 24
plt.figure()
plt.imshow(train_images[index])
plt.colorbar()
plt.grid(False)
plt.show()

print(class_names[train_labels[index][0]])
"""

# Normalization of train and tests sets
train_images = train_images / 255.0

# Definition of a neural network with 3 layers and 2 activation functions
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(32,32,3)),
                          keras.layers.Dense(256, activation="sigmoid"),
                          keras.layers.Dense(10, activation="softmax")
])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Start training
model.fit(train_images, train_labels, epochs=10)

# Check the model performance
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

predictions = model.predict(test_images)
print(predictions)


def check_prediction(index):
    plt.figure()
    plt.imshow(test_images[index])
    plt.grid(False)
    predicted_label = np.argmax(predictions[index])
    prob = int(predictions[index][predicted_label] * 100)
    predicted_class = class_names[predicted_label]
    real_class = class_names[test_labels[index][0]]
    label=predicted_class + " " + str(prob) + "% (" + real_class + ")"
    plt.show()

    if predicted_class == real_class:
        print("\033[0;34;47m " + label)
    else:
        print("\033[0;31;47m " + label)


check_prediction(3)