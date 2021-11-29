import numpy as np
import tensorflow as tf
import constants

class Model(tf.keras.Model):

    def __init__(self, output_size):
        """
        Method to initialize the model

        :param: output_size: the size of the output from the final dense layer ([N,1] where N is the number of days of stock data we are trying to predict)
        :param
        :returns: none
        """
        super(Model, self).__init__()
        # Create out optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Set number of epochs
        self.num_epochs = 16

        # Batch size
        self.batch_size = 2000
        
        # Size of output size of RNN
        self.rnn_size = 128

        # Layers
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size))
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_size, activation="softmax")
        
    def call(self, inputs):
        """
        Use our layers to predict output

        :param : 
        :returns:
        """
        lstm_out = self.lstm(inputs)
        dense1_out = self.dense1(lstm_out)
        dense2_out = self.dense2(dense1_out)

        dense3_out = self.dense3(dense2_out)
        
        return dense3_out
        
    def accuracy(self, outputs, labels):
        """
        Calculate the accuracy of our model

        :param :
        :returns: 
        """
        acc_func = tf.keras.metrics.Accuracy()
        acc_func.update_state(labels, outputs)
        acc = acc_func.result().numpy()

        return acc

    def loss(self, outputs, labels):
        """
        Calculate the loss of our model
        :param :
        :returns:
        """
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        return loss_func(labels, outputs)