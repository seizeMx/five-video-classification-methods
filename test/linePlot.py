from dnn import frame_generator
import matplotlib.pyplot as plt
import random
import numpy as np
from keras.models import load_model
import tensorflow as tf


def check():
    x_scale = 1920 #max_len
    y_scale = 1080 #max_len
    while 1:
        model = load_model('data/checkpoints/dnn-fix-layer-200-600-200.001-0.087-0.004.hdf5', custom_objects={'huber_loss': tf.losses.huber_loss})
        batch_size = 1
        datas = next(frame_generator(batch_size, True))
        max_len = datas[2]
        # inputs = datas[0]
        # lables = datas[1]
        inputs = datas[0]#*max_len
        lables = datas[1]#*max_len

        predictions = model.predict(inputs)

        # max_len = max(predictions.max(), inputs.max())
        # min_len = min(predictions.min(), inputs.min())

        for i in range(0, inputs.shape[1], 20):
            # if i != 160:
            #     continue
            # new_inputs = inputs.reshape([batch_size, batch_size, 10, 10, 2])
            x = inputs[0][i:i + 20:2] * x_scale
            y = inputs[0][i + 1:i + 20:2] * y_scale
            plt.plot(x, y, 'k.', color=(random.random(), random.random(), random.random()), lw=lw, label='frame_%d' % i)  # 'darkorange'

            # x_hat = a[1][0][i:i+10]
            # y_hat = a[1][0][i+10:i+20]
            # plt.plot(x_hat, y_hat, color=(random.random(),random.random(),random.random()), lw=lw, label='y_hat')  #'navy'
            plt.xlim([0.0, x_scale])
            plt.ylim([0.0, y_scale])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            plt.title('input')
            plt.legend(loc="lower right")
        plt.show()

        for i in range(0, inputs.shape[1], 20):
            # x = a[0][0][i:i+10]
            # y = a[0][0][i+10:i+20]
            # plt.plot(x, y, color=(random.random(),random.random(),random.random()), lw=lw, label='y')  #'darkorange'

            x_hat = lables[0][i:i + 10] * x_scale
            y_hat = lables[0][i + 10:i + 20] * y_scale
            plt.plot(x_hat, y_hat, 'ko--', color=(random.random(), random.random(), random.random()), lw=lw, label='lable_%d'%i)  # 'navy'

            # x_hat = predictions[0][i:i + 10] * x_scale
            # y_hat = predictions[0][i + 10:i + 20] * y_scale
            # plt.plot(x_hat, y_hat, color='navy', lw=lw, label='y_hat')  #'darkorange'

            plt.xlim([0.0, x_scale])
            plt.ylim([0.0, y_scale])
            # # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            plt.title('y_lable')
            plt.legend(loc="lower right")
        plt.show()

        for i in range(0, inputs.shape[1], 20):
            # x = a[0][0][i:i+10]
            # y = a[0][0][i+10:i+20]
            # plt.plot(x, y, color=(random.random(),random.random(),random.random()), lw=lw, label='y')  #'darkorange'

            x_hat = lables[0][i:i + 10] * x_scale
            y_hat = lables[0][i + 10:i + 20] * y_scale
            plt.plot(x_hat, y_hat, 'ko--', color='darkorange', lw=lw, label='lable_%d'%i)  # 'navy' (random.random(), random.random(), random.random())

            x_hat = predictions[0][i:i + 10] * x_scale
            y_hat = predictions[0][i + 10:i + 20] * y_scale
            plt.plot(x_hat, y_hat, 'ko--', color='navy', lw=lw, label='y_hat')  #

            plt.xlim([0.0, x_scale])
            plt.ylim([0.0, y_scale])
            # # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            plt.title('y_lable')
            plt.legend(loc="lower right")
        plt.show()

        # for i in range(0, len(predictions[0]), 20):
        #     # x = a[0][0][i:i+10]
        #     # y = a[0][0][i+10:i+20]
        #     # plt.plot(x, y, color=(random.random(),random.random(),random.random()), lw=lw, label='y')  #'darkorange'
        #
        #     x_hat = predictions[1][i:i + 10]
        #     y_hat = predictions[1][i + 10:i + 20]
        #     plt.plot(x_hat, y_hat, color=(random.random(), random.random(), random.random()), lw=lw, label='y_hat')  # 'navy'
        #     plt.xlim([0.0, max_len])
        #     plt.ylim([0.0, max_len])
        #     # plt.xlabel('False Positive Rate')
        #     # plt.ylabel('True Positive Rate')
        #     plt.title('y_hat')
        #     plt.legend(loc="lower right")
        # plt.show()


# def getY(x, k0, k1, k2, k3, k4, k5):
#     return k0 + k1 * (x + k0) + k2 * (x + k0) ** 2 + k3 * (x + k0) ** 3 + k4 * (x + k0) ** 4 + k5 * (x + k0) ** 5


def test():
    xlen = 10000
    xline = np.arange(xlen * (-1), xlen)

    while 1:
        k1 = random.random()
        k2 = random.random()
        k3 = random.random()
        k4 = random.random()
        k5 = random.random()

        k0 = random.randrange(-500, 500)
        # k1 = random.random() * random.randrange(-89, 89)
        # k2 = random.random() * random.randrange(-89, 89) * random.randrange(0, 2)
        # k3 = random.random() * random.randrange(-89, 89) * random.randrange(0, 2)
        # k4 = random.random() * random.randrange(-89, 89) * random.randrange(0, 2)
        # k5 = random.random() * random.randrange(-89, 89) * random.randrange(0, 2)
        k1 = random.random() * random.randrange(-89, 89)  # * random.randint(0, 1)
        k2 = random.random() * random.randrange(-89, 89)  # * random.randint(0, 1)
        k3 = random.random() * random.randrange(-89, 89)  # * random.randint(0, 1)
        k4 = random.random() * random.randrange(-89, 89)  # * random.randint(0, 1)
        k5 = random.random() * random.randrange(-89, 89) * random.randint(0, 1)
        # yline = getY(xline, 0.01, 0.01, 0.01, 0.01, 0, 0) / 100000
        # yline = getY(xline, k0, k1, k2, 0, 0, 0)
        yline = xline  # getY(xline, k0, k1, k2, k3, k4, 0)#TODO
        # log(x), sqrt(x), sin(x), 1 / x, x ^ 2, x ^ 3, x ^ 4, x ^ 5
        plt.plot(xline, yline, color=(random.random(), random.random(), random.random()), lw=lw, label='yline')
        # plt.title('k0 %0.4f, k1 %0.4f, k2 %0.4f, k3 %0.4f, k4 %0.4f, k5 %0.4f' % (k0, k1, k2, k3, k4, k5))
        # plt.title('k0 %0.4f,k1 %0.4f' % (k0 ,k1))
        max_len = max(yline.max(), xline.max())
        min_len = min(xline.min(), yline.min())
        # max_len = 1000  #max(yline.max(), xline.max())
        # min_len = -1000  #min(xline.min(), yline.min())

        # plt.xlim([-10000, 10000])
        # plt.xlim([xline.min(), xline.max()])
        # plt.ylim([yline.min(), yline.max()])
        # plt.xlim([min_len, max_len])
        # plt.ylim([min_len, max_len])

        plt.xlim([-10000, 10000])
        plt.ylim([-10000, 10000])

        plt.show()


if __name__ == "__main__":
    lw = 2
    plt.figure()

    # test()
    check()
    # x_data = [2, 0, 1, 3, 4, 5, 6]
    # y_data = [71000, 58000, 60200, 63000, 84000, 90500, 107000]
    # # 第一个列表代表横坐标的值，第二个代表纵坐标的值
    # plt.plot(x_data, y_data)
    # # 调用show()函数显示图形
    # plt.show()
