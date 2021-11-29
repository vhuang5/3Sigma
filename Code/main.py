import argparse
from preprocessing import get_data
import tensorflow as tf
import constants
import lstm
from visualizations import viz_accuracy, viz_loss

def parse_args():
    """
    Argparser to allow to run different things if needed
    """
    parser = argparse.ArgumentParser(description='3Sigma Final Project')
    parser.add_argument("--lstm",
                        help="use an LSTM model",
                        action='store_true')
    return parser.parse_args()

def train(model, train_commodities, train_stock):
    """
    Runs through x epochs - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """

    for i in range(0,len(train_stock), model.batch_size):

        with tf.GradientTape() as tape:
            pred = model(train_commodities)
            loss = model.loss(pred, train_stock)
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_commodities, test_stock):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    return loss, accuracy

def main():
    args = parse_args()
    if args.lstm:
        model = lstm
    train_stock, train_commodities, test_stock, test_commmodities = get_data(constants.stock_filepath, constants.commodities_filepaths)
    accuracy_per_epoch = []
    loss_per_epoch = []
    for i in range(constants.EPOCH):
        loss = train(model, train_commodities, train_stock)
        print(f"Train Loss for EPOCH {i}: {loss}")
        test_loss, accuracy = test(model, test_commmodities, test_stock)
        print(f"Test Loss for EPOCH {i}: {test_loss}")
        print(f"Test Loss for EPOCH {i}: {accuracy}")
        accuracy_per_epoch += accuracy
        loss_per_epoch += test_loss
    viz_loss(loss_per_epoch)
    viz_accuracy(accuracy_per_epoch)
    print(f"Accuracy after {constants.EPOCH} epochs: {accuracy}")
    print(model.summary())

if __name__ == "__main__":
    main()
