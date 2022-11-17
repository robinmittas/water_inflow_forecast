from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, LSTM, RepeatVector, TimeDistributed, Dropout
# import sys
from ResCNN_LSTM_Layers import ResCNN_Layers


class ResCNN_LSTM_Model(keras.Model):
    '''
    To create an object of this class we need to define following params:
    -------------
    filter1 (int)       -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the first layers block
    filter2 (int)       -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the second layers block
    n_nodes1 (int)      -- the dimensionality of the output space (i.e. the number of output filters in the convolution) for the third layers block
    n_nodes2 (int)      -- the dimensionality of the output space of first Dense Layer
    kernel_size (int)   -- specifies the length of the 1D convolution window.
    n_outputs (int)     -- How many weeks we want to predict: Shape of y_train (target) -- for US Data: 15 Weeks prediction (with 20 weeks input)
    weight_regularizer (int)  -- every coefficient in the weight matrix of the layer will add weight_regularizer * weight_coefficient_value**2 to the total loss of the network
    -------------
    '''

    def __init__(self, filter1, filter2, n_nodes1, n_nodes2, kernel_size, n_outputs, weight_regularizer, use_cnn_layers,
                 use_lstm_layers, dropout_probability=0, use_batch_normalization=False):
        super(ResCNN_LSTM_Model, self).__init__()
        self.dropout_probability = dropout_probability
        self.use_cnn_layers = use_cnn_layers
        if use_cnn_layers[0]:
            self.block1 = ResCNN_Layers(filter1, kernel_size, weight_regularizer, use_batch_normalization)
        if use_cnn_layers[1]:
            self.block2 = ResCNN_Layers(filter2, kernel_size, weight_regularizer, use_batch_normalization)
        if use_cnn_layers[2]:
            self.block3 = ResCNN_Layers(n_nodes1, kernel_size, weight_regularizer, use_batch_normalization)
        self.flatten = Flatten()
        self.repeat_vector = RepeatVector(n_outputs)
        self.lstm1 = LSTM(n_nodes1, activation='relu', kernel_regularizer=regularizers.l2(weight_regularizer),
                          return_sequences=True, dropout=dropout_probability, recurrent_dropout=dropout_probability)
        self.lstm2 = LSTM(n_nodes1, activation='relu', kernel_regularizer=regularizers.l2(weight_regularizer),
                          return_sequences=True, dropout=dropout_probability, recurrent_dropout=dropout_probability)
        self.lstm3 = LSTM(n_nodes1, activation='relu', kernel_regularizer=regularizers.l2(weight_regularizer),
                          return_sequences=True, dropout=dropout_probability, recurrent_dropout=dropout_probability)
        self.lstm4 = LSTM(n_nodes1, activation='relu', kernel_regularizer=regularizers.l2(weight_regularizer),
                          return_sequences=True, dropout=dropout_probability, recurrent_dropout=dropout_probability)
        self.dense = TimeDistributed(Dense(n_nodes2, activation='relu', kernel_regularizer=regularizers.l2(weight_regularizer)))
        if dropout_probability > 0:
            self.dense_dropout = Dropout(dropout_probability)
        self.output_tensor = TimeDistributed(Dense(1))
        self.use_lstm_layers = use_lstm_layers


    def call(self, input_tensor):
        if self.use_cnn_layers[0]:
            x = self.block1(input_tensor)
        else:
            x = input_tensor
        if self.use_cnn_layers[1]:
            x = self.block2(x)
        if self.use_cnn_layers[2]:
            x = self.block3(x)

        x = self.flatten(x)
        x = self.repeat_vector(x)

        if self.use_lstm_layers[0]:
            x = self.lstm1(x)
        if self.use_lstm_layers[1]:
            x = self.lstm2(x)
        if self.use_lstm_layers[2]:
            x = self.lstm3(x)
        if self.use_lstm_layers[3]:
            x = self.lstm4(x)

        dense = self.dense(x)
        if self.dropout_probability > 0:
            dense = self.dense_dropout(dense)
        output = self.output_tensor(dense)

        return output
