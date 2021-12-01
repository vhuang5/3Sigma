import numpy as np
import tensorflow as tf
import constants

class LSTM(tf.keras.Model):

    def __init__(self, output_size):
        """
        Method to initialize the model

        :param: output_size: the size of the output from the final dense layer ([N,1] where N is the number of days of stock data we are trying to predict)
        :param
        :returns: none
        """
        super(LSTM, self).__init__()
        # Create out optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Batch size
        self.batch_size = 50
        
        # Size of output size of LSTM
        self.lstm_size = 64

        # Layers
        self.lstm = tf.keras.layers.LSTM(self.lstm_size, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_size, activation='relu')
        
    def call(self, inputs, initial_state=None):
        """
        Use our layers to predict output

        :param: inputs: (N,5) matrix of commodities data
        :returns: (N,1) matix
        """
        lstm_out, state_h, state_c = self.lstm(inputs, initial_state = initial_state)
        dense1_out = self.dense1(lstm_out)
        dense2_out = self.dense2(dense1_out)

        dense3_out = self.dense3(dense2_out)
        # print(dense3_out.shape)
        return dense3_out
        
    def accuracy(self, outputs, labels):
        """
        Calculate the accuracy of our model

        :param outputs: predictions from the model
        :param labels: (N,1) matrix of closing stock prices
        :returns: accuracy of the batch
        """
        acc_func = tf.keras.metrics.Accuracy()
        acc_func.update_state(labels, outputs)
        acc = acc_func.result().numpy()

        return acc

    def loss(self, outputs, labels):
        """
        Calculate the loss of our model
        :param outputs: predictions from the model
        :param labels: (N,1) matrix of closing stock prices
        :returns: loss of the batch
        """
        loss_func = tf.keras.losses.MeanAbsoluteError()
        return tf.reduce_mean(loss_func(labels, outputs))