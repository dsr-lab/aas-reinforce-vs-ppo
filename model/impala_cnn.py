import tensorflow as tf


class ImpalaNet(tf.keras.Model):

    def __init__(self, depths=[16, 32, 64]):

        super(ImpalaNet, self).__init__()

        self.convolutional_blocks = \
            [ConvolutionalBlock(depth) for depth in depths]

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='relu')

    def call(self, x):
        output = x
        for conv_block in self.convolutional_blocks:
            output = conv_block(output)

        output = self.flatten(output)
        output = tf.keras.layers.ReLU()(output)
        output = self.dense(output)

        return output


class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, n_channels, kernel_size=3):
        super(ConvolutionalBlock, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(n_channels, kernel_size, padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.residual_block_1 = ResidualBlock(n_channels)
        self.residual_block_2 = ResidualBlock(n_channels)

    def call(self, x):
        output = self.conv_layer(x)
        output = self.max_pool(output)
        output = self.residual_block_1(output)
        output = self.residual_block_2(output)

        return output


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, n_channels, kernel_size=3, n_convolutions=2):
        super(ResidualBlock, self).__init__()

        self.conv_layers = \
            [tf.keras.layers.Conv2D(n_channels, kernel_size, padding='same') for i in range(n_convolutions)]

    def call(self, x):
        output = x
        for layer in self.conv_layers:
            output = tf.keras.layers.ReLU()(output)
            output = layer(output)

        return output + x


# a = ResidualBlock(1)
# b = np.random.rand(1, 15, 15, 3)
# a(b)

# b = np.random.rand(1, 15, 15, 3)
# model = ImpalaNet()
# result = model(b)
# model.summary()
# print()

