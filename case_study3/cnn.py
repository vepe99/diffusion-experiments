import bayesflow as bf
import keras


@bf.utils.serialization.serializable("custom")
class ResNetSubnet(bf.networks.SummaryNetwork):
    def __init__(
            self,
            widths=(8, 16, 32),
            activation="mish",
            **kwargs,
    ):
        super().__init__(**kwargs)

        layers = [keras.layers.Conv2D(width, kernel_size=3, activation=activation, padding='SAME') for width in widths]
        self.net = bf.networks.Sequential(layers)
        self.last_channels = widths[-1]

    def build(self, x_shape, t_shape, conditions_shape):
        self.net.build(x_shape[:-1] + [4,])

    def call(self, x, t, conditions, training=False):
        t = keras.ops.broadcast_to(t, keras.ops.shape(x)[:-1] + (1,))
        return self.net(keras.ops.concatenate((x, t, conditions), axis=-1), training=training)

    def compute_output_shape(self, x_shape, t_shape, conditions_shape):
        return x_shape[:-1] + [self.last_channels,]