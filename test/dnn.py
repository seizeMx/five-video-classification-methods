import os
import time
import random

from keras import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF

from data import threadsafe_generator


def process(x, k0, k1, k2,):
    # return k0 + k1 * (x+k0) + k2 * (x+k0) ** 2 + k3 * (x+k0) ** 3 + k4 * (x+k0) ** 4 + k5 * (x+k0) ** 5
    # s11 = (-1 + random.randint(0, 1))
    choice = random.random()
    if choice > 0.3:
        return k1 * x + k2 * (x + random.randint(-1080, 1080)) ^ 2 + k0 * random.randint(-1920, 1920)
    else:
        return k1 * (x + random.randint(-1080, 1080)) + k0 * random.randint(-1920, 1920)

def getY(x, k0, k1, k2, k3, k4, k5):
    return k0 + k1 * (x + k0) + k2 * (x + k0) ** 2 + k3 * (x + k0) ** 3 + k4 * (x + k0) ** 4 + k5 * (x + k0) ** 5


@threadsafe_generator
def frame_generator(batch_size):
    """Return a generator that we can use to train on. There are
    a couple different things we can return:

    data_type: 'gen', 'file'

    """
    # Get the right dataset for the generator.
    # train, test = self.split_train_test()
    while 1:
        # b_tensor_lable = np.zeros((batch_size, 10, 10, 2))
        b_tensor_input = np.zeros((batch_size, 10, 10, 2))

        b_xy_lable = np.zeros((batch_size, 10, 20))
        # b_xy_input = np.zeros((batch_size, 10, 20))
        for b in range(0, batch_size):

            # xy_lable = []
            # xy_input = []
            ran = random.randint(2, 5)
            for i in range(0, 10):
                k0 = random.random()
                k1 = random.random()
                k2 = random.random()
                k3 = random.random()
                k4 = random.random()
                k5 = random.random()

                # y = getY(x, k0, k1, k2 if i in range(2, 10) else 0, k3 if i in range(4, 10) else 0, k4 if i in range(6, 10) else 0, k5 if i in range(8, 10) else 0)
                ran_5 = 0  # random.randint(0, 1)
                if ran_5 == 0:
                    if i in (2, 3, 4, 5):
                        # x = np.random.rand(10)
                        rangeLen = random.randint(50, 1080)
                        x = np.random.rand(10) * rangeLen + np.random.randint(1, 1080 - rangeLen)
                        # x = np.random.rand(10) * np.random.randint(1, 1080)
                        x.sort()

                        y = getY(x, k0, k1, k2 if i == k2 else 0, k3 if i == k3 else 0, k4 if i == k4 else 0, k5 if i == k5 else 0)
                        # y = getY(x, k0, k1, k2, k3, 0, 0)
                    else:
                        # x = np.random.rand(10)
                        rangeLen = random.randint(50, 100)
                        x = np.random.rand(10) * rangeLen + np.random.randint(1, 1080 - rangeLen)
                        # x = np.random.rand(10) * np.random.randint(1, 1080)
                        x.sort()

                        y = getY(x, k0, k1, k2 if ran == k2 else 0, k3 if ran == k3 else 0,
                                 k4 if ran == k4 else 0, k5 if ran == k5 else 0)
                else:
                    x = np.random.rand(10)
                    y = getY(x, k0, k1, k2, k3, k4, k5)
                # Normalize
                # t_min = min(x.min(), y.min())
                # t_max = max(x.max(), y.max())
                # x = x/t_max
                # y = y/t_max

                b_tensor_input[b, i, :, 0] = x
                b_tensor_input[b, i, :, 1] = y
                # # shuffle exclude first one
                # points = list(zip(x, y))[1:]
                # random.shuffle(points)  # TODO np.random.shuffle() instead
                # points = list(zip(*points))
                # # xy_input.append(points[0])
                # # xy_input.append(points[1])
                #
                # b_xy_input[b, i, 0] = np.asarray(x[0])
                # b_xy_input[b, i, 10] = np.asarray(y[0])
                # b_xy_input[b, i, 1:10] = np.asarray(points[0])
                # b_xy_input[b, i, 11:20] = np.asarray(points[1])
                #
                # # xy_lable.append(np.array(x))
                # # xy_lable.append(np.array(y))

                b_xy_lable[b, i, 0:10] = x[:]
                b_xy_lable[b, i, 10:20] = y[:]

            # shuffle frames exclude first frame
            for j in range(1, 10):
                np.random.shuffle(b_tensor_input[b, :, j, :])

        # b_xy_input = b_xy_input/b_xy_input.max()
        yield b_tensor_input.swapaxes(1, 2).reshape((b + 1, 200)), b_xy_lable.reshape((b + 1, 200))


def dnn(saved_weights=None):
    model = Sequential()

    # 1 Total params: 200,800
    # model.add(Dense(200, activation='relu', input_shape=(200,)))
    # model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(200, activation='relu'))

    # 2 Total params: 281,000
    model.add(Dense(200, activation='relu', input_shape=(200,)))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))

    # 3 Total params: 361,200
    # model.add(Dense(200, activation='relu', input_shape=(200,)))
    # model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(400, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(200, activation='relu'))

    if saved_weights is not None:
        print('load weights ', saved_weights)
        model.load_weights(saved_weights)

    # Now compile the network.
    optimizer = Adam(lr=5e-5, decay=1e-3)
    model.compile(loss=tf.losses.huber_loss, optimizer=optimizer,
                  metrics=['accuracy'])

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
    early_stopper = EarlyStopping(patience=5)

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
