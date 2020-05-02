import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, in_shape, num_hidden_layers, num_hidden_units, out_shape):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(in_shape, ))  # input layer
        self.hidden_layers = []  # hidden layers
        for i in range(num_hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(num_hidden_units[i], activation='tanh', kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2))
        self.output_layer = tf.keras.layers.Dense(out_shape, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, x):
        # returns result of forward pass for x
        res = self.input_layer(x)
        for layer in self.hidden_layers:
            res = layer(res)
        res = self.output_layer(res)
        return res
