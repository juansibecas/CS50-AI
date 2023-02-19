import cv2
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split

EPOCHS = 25
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    for n_cat in range(NUM_CATEGORIES):
        folder_dir = os.path.join(data_dir, str(n_cat))
        image_names = os.listdir(folder_dir)
        for name in image_names:
            img_dir = os.path.join(folder_dir, name)
            img = cv2.imread(img_dir)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            images.append(img)
            labels.append(n_cat)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model_settings = {
        "dropout": True,
        "p_dropout": 0.2,
        "npl": 100,  # neurons per layer
        "hidden_activation": "relu",
        "output_activation": "softmax",
        "conv_and_pool": 3,  # how many times conv2d and pooling are called
        "conv_activation": "relu",
        "conv_filters": 32,  # doubling for each conv2d
        "kernel_size": (3, 3),
        "pool_size": (2, 2),
        "optimizer": "adam",
        "loss": "categorical_crossentropy"
    }

    dropout = model_settings["dropout"]
    p_dropout = model_settings["p_dropout"]
    npl = model_settings["npl"]
    hidden_activation = model_settings["hidden_activation"]
    output_activation = model_settings["output_activation"]
    conv_and_pool = model_settings["conv_and_pool"]
    conv_activation = model_settings["conv_activation"]
    conv_filters = model_settings["conv_filters"]
    kernel_size = model_settings["kernel_size"]
    pool_size = model_settings["pool_size"]
    optimizer = model_settings["optimizer"]
    loss = model_settings["loss"]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1./255, offset=0, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.RandomFlip(mode="horizontal"),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Conv2D(conv_filters, kernel_size, activation=conv_activation),
            tf.keras.layers.MaxPooling2D(pool_size),
            tf.keras.layers.Conv2D(conv_filters, kernel_size, activation=conv_activation),
            tf.keras.layers.MaxPooling2D(pool_size),
            tf.keras.layers.Conv2D(conv_filters, kernel_size, activation=conv_activation),
            tf.keras.layers.MaxPooling2D(pool_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(npl, activation=hidden_activation),
            tf.keras.layers.Dropout(p_dropout),
            tf.keras.layers.Dense(npl, activation=hidden_activation),
            tf.keras.layers.Dropout(p_dropout),
            tf.keras.layers.Dense(npl, activation=hidden_activation),
            tf.keras.layers.Dropout(p_dropout),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation=output_activation)

        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()
