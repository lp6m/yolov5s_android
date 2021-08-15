import tensorflow as tf
from tensorflow import keras
import numpy as np

class tf_Detect(keras.layers.Layer):
    # def __init__(self, nc=80, anchors=(), ch=(), w=None):  # detection layer
    
    def __init__(self, nc=80):
        anchorgrid = np.array([
            [10, 13, 16, 30, 33, 23], #80
            [30, 61, 62, 45, 59, 119], #40
            [116, 90, 156, 198, 373, 326] #20
        ])
        strides =  np.array([8, 16, 32])
        self.IMAGE_SIZE_H = 640
        self.IMAGE_SIZE_W = 640
        super(tf_Detect, self).__init__()
        self.stride = tf.convert_to_tensor(strides, dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchorgrid)  # number of detection layers 3
        self.na = len(anchorgrid[0]) // 2  # number of anchors 3
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchor_grid = tf.reshape(tf.convert_to_tensor(anchorgrid, dtype=tf.float32),
                                      [self.nl, 1, -1, 1, 2])
        for i in range(self.nl):
            ny, nx = self.IMAGE_SIZE_H // self.stride[i], self.IMAGE_SIZE_W // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        # x = x.copy()  # for profiling
        z = []  # inference output
        x = []
        for i in range(self.nl):
            # x.append(self.m[i](inputs[i]))
            print(inputs[i].shape)
            x.append(inputs[i])
            # Reshape: (bs,20,20,255) to (bs,20*20, 3, 85)
            # Transpose (bs, 20*20, 3, 85) to (bs, 3, 20*20, 85)
            ny, nx = self.IMAGE_SIZE_H // self.stride[i], self.IMAGE_SIZE_W // self.stride[i]
            x[i] = tf.transpose(tf.reshape(x[i], [-1, ny * nx, self.na, self.no]), [0, 2, 1, 3])

            y = tf.sigmoid(x[i])
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            # Normalize xywh to 0-1 to reduce calibration error
            xy /= tf.constant([[self.IMAGE_SIZE_W, self.IMAGE_SIZE_H]], dtype=tf.float32)
            wh /= tf.constant([[self.IMAGE_SIZE_W, self.IMAGE_SIZE_H]], dtype=tf.float32)
            y = tf.concat([xy, wh, y[..., 4:]], -1)
            z.append(tf.reshape(y, [-1, 3 * ny * nx, self.no]))

        return tf.concat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)