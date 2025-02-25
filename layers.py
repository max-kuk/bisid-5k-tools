from keras import layers, ops, saving


@saving.register_keras_serializable()
class MinMaxScaler(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        min_vals = ops.min(inputs, keepdims=True, axis=-1)
        max_vals = ops.max(inputs, keepdims=True, axis=-1)

        # Ensure that the range is not zero to avoid division by zero
        range_nonzero = ops.where(min_vals != max_vals, max_vals - min_vals, 1.0)

        # Normalize each pixel by subtracting the minimum and dividing by the range
        output = (inputs - min_vals) / range_nonzero

        return output
