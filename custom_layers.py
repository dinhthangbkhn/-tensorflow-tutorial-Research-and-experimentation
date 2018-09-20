import tensorflow as tf

tfe = tf.contrib.eager
tf.enable_eager_execution()
layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))


# print(layer(tf.zeros([10,5])))
# print(layer.variables)
# print(layer.kernel, layer.bias)


# ------------------------Implementing custom layers------------------------------------
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        """
        Set up the output for layer
        :param num_outputs:
        """
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        """
        Create a Variable that has dimension [last dimension of input, num_outputs ]
        :param input_shape:
        :return:
        """
        self.kernel = self.add_variable("kernel", shape=[input_shape[-1].value, self.num_outputs])

    def call(self, input):
        """
        Calculate matrix multiply between input and layer variable
        :param input:
        :return:
        """
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(5)

print(layer)
print(layer(tf.ones([10, 5])))
print(layer.variables)
print(layer.kernel)


# -----------------------------------------Models: composing layers --------------------------------

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        x = tf.nn.relu(x)
        return x


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x for x in block.variables])
