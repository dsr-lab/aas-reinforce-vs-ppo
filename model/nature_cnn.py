import tensorflow as tf


class NatureNet(tf.keras.Model):
    """
    Class used for creating the Nature-CNN backbone
    """
    def __init__(self):

        super(NatureNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(64, 64, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='relu')

    def call(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)

        output = self.flatten(output)
        output = self.dense(output)

        return output
