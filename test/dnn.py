import os
import time
import random

from keras import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import Lambda

from data import threadsafe_generator

def process(x, k0, k1, k2):
    # return k0 + k1 * (x+k0) + k2 * (x+k0) ** 2 + k3 * (x+k0) ** 3 + k4 * (x+k0) ** 4 + k5 * (x+k0) ** 5
    # s11 = (-1 + random.randint(0, 1))
    choice = random.random()
    # if choice > 0.8:
    #     return k3 * (x + random.randint(-1080, 1080)) ** 3 + k0 * random.randint(-1920, 1920)
    # el
    if choice > 0.1:
        return k1 * x + k2 * (x + random.randint(-1920, 1920)) ** 2 + k0 * random.randint(-1080, 1080)
    else:
        return k1 * (x + random.randint(-1920, 1920)) + k0 * random.randint(-1080, 1080)


# def getY(x, k0, k1, k2, k3, k4, k5):
#     return k0 + k1 * (x + k0) + k2 * (x + k0) ** 2 + k3 * (x + k0) ** 3 + k4 * (x + k0) ** 4 + k5 * (x + k0) ** 5


@threadsafe_generator
def frame_generator(batch_size, withMax=False):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:

    data_type: 'gen', 'file'

    """
    # Get the right dataset for the generator.
    # train, test = self.split_train_test()
    temp_max = 1
    while 1:
        b_tensor_input = np.zeros((batch_size, 10, 10, 3))

        b_xy_lable = np.zeros((batch_size, 10, 20))
        for b in range(0, batch_size):

            for person_i in range(0, 10):
                k0 = random.random() * (-1 if random.randint(0, 1) == 0 else 1)
                k1 = random.random() * (-1 if random.randint(0, 1) == 0 else 1)
                k2 = random.random() * (-1 if random.randint(0, 1) == 0 else 1)

                # x = np.random.rand(10)
                # rangeLen = random.randint(50, 1920-1)
                # x = np.random.rand(10) * rangeLen + np.random.randint(0, 1920 - rangeLen)
                # x = np.random.rand(10) * np.random.randint(100, 1080)
                p1_start = np.random.randint(0, 150)
                p1_stop = np.random.randint(p1_start + 1, 1080 - 1)
                x = np.arange(p1_start, p1_stop, (p1_stop - p1_start) / 10)

                x.sort()
                y = process(x, k0, k1, k2)

                # Normalize
                # t_min = min(x.min(), y.min())
                # t_max = max(x.max(), y.max())

                x = x / (x.max() + 1)
                if y.min() < 0:
                    y_minus = y.min() - (y.max() - y.min()) * np.random.rand()
                else:
                    y_minus = y.min() - y.min() * np.random.rand()

                # y_minus = np.random.randint(min(y.min(), y_minus), max(y.min(), y_minus))
                y = (y - y_minus) / (y.max() - y_minus + 1)
                #
                # b_temp

                b_tensor_input[b, person_i, :, 0] = x
                b_tensor_input[b, person_i, :, 1] = y
                b_tensor_input[b, person_i, :, 2] = np.arange(0, 10)

                # b_xy_lable[b, person_i, 0:10] = x[:]
                # b_xy_lable[b, person_i, 10:20] = y[:]

            # shuffle frames exclude first frame
            for frame_j in range(1, 10):
                np.random.shuffle(b_tensor_input[b, :, frame_j, :])
        # temp_min = b_xy_lable.min()
        # temp_max = b_xy_lable.max()-temp_min
        # b_xy_lable = (b_xy_lable-temp_min)/temp_max
        # b_tensor_input = (b_tensor_input-temp_min)/temp_max
        # temp_max = b_xy_lable.max()
        # b_xy_lable = b_xy_lable / temp_max
        # b_tensor_input = b_tensor_input / temp_max
        if withMax:
            yield b_tensor_input[:, :, :, 0:2].swapaxes(1, 2).reshape((b + 1, 200)), b_tensor_input[:, :, :, 2].swapaxes(1, 2).reshape((b + 1, 100)).astype(np.int32), temp_max
        else:
            yield b_tensor_input[:, :, :, 0:2].swapaxes(1, 2).reshape((b + 1, 200)), b_tensor_input[:, :, :, 2].swapaxes(1, 2).reshape((b + 1, 100)).astype(np.int32)

def top_k(input, k):
  # Can also use `.values` to return a sorted tensor
  return tf.cast(tf.reshape(tf.nn.top_k(tf.reshape(input, (-1, 10, 10)), k=k, sorted=True).indices, (-1, 100)), dtype=tf.int32)

def mycrossentropy(y_true, y_pred):
    # print('y_pred.shape()', y_pred.shape())
    y_pred_process = KTF.reshape(y_pred, (-1, 10, 10))
    y_pred_process = top_k(y_pred_process, 10)
    # y_pred_process = K.argsort(y_pred_process, axis=-1)
    y_pred_process = KTF.reshape(y_pred_process, (-1, 100))
    return K.categorical_crossentropy(tf.cast(y_true, dtype=tf.int32), tf.cast(y_pred_process, dtype=tf.int32))

def dnn(saved_weights=None):
    # act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)

    # input = Input(shape=(200,))
    #
    # x = Dense(200)(input)
    # x = LeakyReLU(alpha=0.01)(x)
    # x = Dense(1200)(x)
    # x = LeakyReLU(alpha=0.01)(x)
    #
    # p0 = Dense(11, activation='softmax')(x)
    # p1 = Dense(11, activation='softmax')(x)
    # p2 = Dense(11, activation='softmax')(x)
    # p3 = Dense(11, activation='softmax')(x)
    # p4 = Dense(11, activation='softmax')(x)
    # p5 = Dense(11, activation='softmax')(x)
    # p6 = Dense(11, activation='softmax')(x)
    # p7 = Dense(11, activation='softmax')(x)
    # p8 = Dense(11, activation='softmax')(x)
    # p9 = Dense(11, activation='softmax')(x)

    # 1 Total params: 200,800
    # model.add(Dense(200, activation='relu', input_shape=(200,)))
    # model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(200, activation='relu'))

    # 2 Total params: 281,000
    model = Sequential()
    # model.add(Dense(200, input_shape=(200,)))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(Dense(1200))
    # model.add(LeakyReLU(alpha=0.01))
    # model.add(Dropout(0.1))
    # model.add(Dense(200))

    # 3 Total params: 361,200
    model.add(Dense(200, input_shape=(200,)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(600))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(800))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(800))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(600))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(100))

    # model.add(Lambda(top_k, input_shape=(-1, 10, 10), arguments={'k': 10}))

    if saved_weights is not None:
        print('load weights ', saved_weights)
        model.load_weights(saved_weights)

    # Now compile the network.
    optimizer = Adam(lr=1e-3, decay=1e-3)
    model.compile(loss=mycrossentropy, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=tf.losses.huber_loss, optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model


def main():
    batch_size = 1024
    model = 'dnn'
    data_type = 'fix-layer-200-600-200'
    epochs = 20
    steps_per_epoch = 10000

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + '.{epoch:03d}-{val_acc:.3f}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=3)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + str(timestamp) + '.log'))

    # Get the data and process it.

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.

    # Get generators.
    generator = frame_generator(batch_size)

    # rm = dnn('data/checkpoints/dnn-gen.002-0.966-188938182656.000.hdf5') 5
    # rm = dnn('data/checkpoints/dnn-gen2.001-0.699-14384.461.hdf5') #5->squre
    # rm = dnn('data/checkpoints/dnn-gen5.001-0.821-52.715.hdf5') #5->squre
    # rm = dnn('data/checkpoints/dnn-gen5-600-.004-0.965-4.943.hdf5') #5->squre
    # rm = dnn('data/checkpoints/dnn-gen5-3layer-200-600-200-.001-0.913-8.235.hdf5')
    rm = dnn()

    # Use fit generator.
    rm.model.fit_generator(
        use_multiprocessing=True,
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=generator,
        validation_steps=200,
        workers=6)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    main()
