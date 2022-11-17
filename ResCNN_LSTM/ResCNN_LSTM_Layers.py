from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, ReLU, Add, BatchNormalization
from tensorflow.keras import regularizers


class ResCNN_Layers(keras.layers.Layer):
    """
    This keras.layers.Layer subclass is used to create the standard block used for Neural Network
    -------------
    out_channels (int)        -- the dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernel_size (int)         -- specifies the length of the 1D convolution window.
    weight_regularizer (int)  -- every coefficient in the weight matrix of the layer will add weight_regularizer * weight_coefficient_value**2 to the total loss of the network
    -------------
    """

    def __init__(self, out_channels, kernel_size, weight_regularizer, use_batch_normalization=False):
        super(ResCNN_Layers, self).__init__()
        self.conv1 = Conv1D(out_channels, kernel_size, padding='causal',
                            kernel_regularizer=regularizers.l2(weight_regularizer))
        self.conv2 = Conv1D(out_channels, kernel_size, padding='causal',
                            kernel_regularizer=regularizers.l2(weight_regularizer))
        self.conv3 = Conv1D(out_channels, kernel_size, padding='causal',
                            kernel_regularizer=regularizers.l2(weight_regularizer))
        self.conv4 = Conv1D(out_channels, kernel_size, padding='causal',
                            kernel_regularizer=regularizers.l2(weight_regularizer))
        self.use_batch_normalization = use_batch_normalization
        if use_batch_normalization:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.bn3 = BatchNormalization()
            self.bn4 = BatchNormalization()
        self.relu = ReLU()
        self.add = Add()
        self.max_pooling = MaxPooling1D()

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.conv1.filters,
            "kernel_size": self.conv1.kernel_size,
        })
        return config

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        if self.use_batch_normalization:
            x = self.bn1(x)
        residual = self.relu(x)
        x = self.conv2(residual)
        x = self.add([residual, x])
        if self.use_batch_normalization:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        if self.use_batch_normalization:
            x = self.bn3(x)
        residual = self.relu(x)
        x = self.conv4(residual)
        x = self.add([residual, x])
        if self.use_batch_normalization:
            x = self.bn4(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        return x
