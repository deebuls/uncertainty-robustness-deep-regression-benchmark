import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential

class DenseNormal(Layer):
    def __init__(self, units):
        super(DenseNormal, self).__init__()
        self.units = int(units)
        self.dense = Dense(2 * self.units)

    def call(self, x):
        output = self.dense(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.units)

    def get_config(self):
        base_config = super(DenseNormal, self).get_config()
        base_config['units'] = self.units
        return base_config


class DenseNormalGamma(Layer):
    def __init__(self, units):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha)+1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4*self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config

class DenseDirichlet(Layer):
    def __init__(self, units):
        super(DenseDirichlet, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        output = self.dense(x)
        evidence = tf.exp(output)
        alpha = evidence + 1
        prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)
        return tf.concat([alpha, prob], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.units)

class DenseSigmoid(Layer):
    def __init__(self, units):
        super(DenseSigmoid, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        logits = self.dense(x)
        prob = tf.nn.sigmoid(logits)
        return [logits, prob]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class Conv2DNormal(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv2DNormal, self).__init__()
        self.conv = tf.keras.layers.Conv2D(2*filters, kernel_size, **kwargs)

    def call(self, x):
        output = self.conv(x)
        mu, logsigma = tf.split(output, 2, axis=-1)
        sigma = tf.nn.softplus(logsigma) + 1e-6
        # return [mu, sigma]
        return tf.concat([mu, sigma], axis=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        base_config = super(Conv2DNormal, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        return base_config



class Conv2DNormalGamma(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(Conv2DNormalGamma, self).__init__()
        self.conv = tf.keras.layers.Conv2D(4*filters, kernel_size, **kwargs)

    def evidence(self, x):
        # return tf.exp(x)
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.conv(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha)+1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        base_config = super(Conv2DNormalGamma, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        return base_config

# Conv2DNormalGamma(32, (5,5))
