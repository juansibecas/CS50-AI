import traffic
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json

EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

optimizer = 'rmsprop'  # 8th loop all optimizers. rmsprop, nadam, adam in order

loss = 'poisson'    # 7th loop all loss_eqs. poisson, mean_square_logarithmic_error, kl_divergence in order

activation = 'softsign'  # 6th loop all hidden activation curves. softsign, swish, tanh, gelu relu elu in order

convolution = 'lstm'  # 4th normal, separable, depthwise, lstm. separable and lstm do better. pick lstm

pooling = 'max'     # 4th max, avg. doubling filters. max does better with more cycles
                    # 5th max, avg. constant filters. max is better

dropout = True      # 1st True False, true might be a little better, can't tell
                    # 2nd True False, true is a little better

p_dropout = 0.2     # 1st 0.1, 0.3, 0.5, best value between 0.1 and 0.3
                    # 2nd 0.15, 0.2, 0.25, 0.2 is better

n_layers = 2        # 1st 1, 3, 5, 7 best value between 1 and 3, picked 2

npl = 250           # 1st 20, 60, 100, 160, 200 best value between 100 and 200
                    # 2nd 100, 120, 140, 160, 200, around 200 is better
                    # 3rd 180, 200, 240, 280, 320, 240 is better

conv_and_pool = 2   # 3rd 1, 2, 3. 2 looks better (only tried max pooling)
                    # 4th 1, 2, 3. doubling filters. 2 with max_pooling seems better
                    # 5th 1, 2, 3. constant filters. 2 is better

conv_filters = 32   # 4th 16 duplicating for each conv
                    # 5th 32 constant. 32 constant is better


def main():
    images, labels = traffic.load_data('gtsrb')

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model_settings = {
        "dropout": dropout,                 # True or False
        "p_dropout": p_dropout,             # probability
        "n_layers": n_layers,
        "npl": npl,                         # neurons per layer
        "hidden_activation": activation,
        "output_activation": "softmax",
        "conv_and_pool": conv_and_pool,   # how many times conv2d and
                                            # pooling are called
        "conv_activation": "relu",
        "conv_filters": conv_filters,
        "kernel_size": (3, 3),
        "pool_size": (2, 2),
        "optimizer": optimizer,
        "loss": loss,
        "conv_method": convolution,
        "pool_method": pooling
    }
    model = get_model(model_settings)

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    l, acc = model.evaluate(x_test, y_test, verbose=2)

    model_name = f'{i} acc{acc}'
    model_dir = os.path.join('models', model_name)
    model.save(model_dir, save_format='tf')
    with open(f'{model_dir}.json', 'w') as fp:
        json.dump(model_settings, fp)


def get_model(settings):
    kernel_size = settings["kernel_size"]
    pool_size = settings["pool_size"]
    hidden_activation = settings["hidden_activation"]
    output_activation = settings["output_activation"]
    conv_activation = settings["conv_activation"]
    conv_method = settings['conv_method']
    pool_method = settings['pool_method']
    n_npl = settings['npl']
    n = settings['n_layers']
    p = settings['p_dropout']
    filters = settings['conv_filters']
    opt = settings['optimizer']
    loss_eq = settings['loss']

    layers = [
        tf.keras.layers.Rescaling(scale=1. / 255, offset=0, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    ]

    for i in range(settings['conv_and_pool']):
        if conv_method == 'normal':
            layers.append(tf.keras.layers.Conv2D(filters, kernel_size, activation=conv_activation))
        elif conv_method == 'lstm':
            layers.append(tf.keras.layers.Conv2D(filters, kernel_size, activation=conv_activation))
        elif conv_method == 'depthwise':
            layers.append(tf.keras.layers.Conv2D(filters, kernel_size, activation=conv_activation))
        elif conv_method == 'separable':
            layers.append(tf.keras.layers.Conv2D(filters, kernel_size, activation=conv_activation))

        if pool_method == 'max':
            layers.append(tf.keras.layers.MaxPooling2D(pool_size))
        elif pool_method == 'avg':
            layers.append(tf.keras.layers.AveragePooling2D(pool_size))

    layers.append(tf.keras.layers.Flatten())

    for i in range(n):
        layers.append(tf.keras.layers.Dense(n_npl, activation=hidden_activation))
        if settings['dropout']:
            layers.append(tf.keras.layers.Dropout(p))

    layers.append(tf.keras.layers.Dense(NUM_CATEGORIES, activation=output_activation))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=opt,
        loss=loss_eq,
        metrics=["accuracy"]
    )

    return model


if __name__ == '__main__':
    main()
