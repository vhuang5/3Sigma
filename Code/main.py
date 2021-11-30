import argparse
from preprocessing import get_data
import tensorflow as tf
import constants
from lstm import LSTM
from mlp import MLP
from visualizations import viz_accuracy, viz_loss

def parse_args():
    """
    Argparser to allow to run different things if needed
    """
    parser = argparse.ArgumentParser(description='3Sigma Final Project')
    parser.add_argument("--lstm",
                        help="use an LSTM model",
                        action='store_true')
    parser.add_argument("--mlp",
                        help="use multi-layer-perceptron model",
                        action='store_true')
    return parser.parse_args()

def train(model, train_commodities, train_stock):
    """
    Runs through x epochs - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,1)
    :return: None
    """
    total_loss = 0
    for i in range(0,len(train_stock), model.batch_size):
        batched_commodities = train_commodities[i:i+model.batch_size, :]
        batched_stock = train_stock[i:i+model.batch_size, :]
        with tf.GradientTape() as tape:
            pred = model.call(batched_commodities)
            loss = model.loss(pred, batched_stock)
            total_loss += loss
            # print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss/model.batch_size

def test(model, test_commodities, test_stock):
    """
    Runs through x epochs - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,1)
    :returns: loss and  of the test set
    """
    total_loss = 0
    total_accuracy = 0
    for i in range(0,len(test_stock), model.batch_size):
        batched_commodities = test_commodities[i:i+model.batch_size, :]
        batched_stock = test_stock[i:i+model.batch_size, :]
        pred = model.call(batched_commodities)
        loss = model.loss(pred, batched_stock)
        total_loss += loss
        acc = model.accuracy(pred, batched_stock)
        total_accuracy += acc
        # print(loss)
    return total_loss/model.batch_size, total_accuracy/model.batch_size

def main():
    args = parse_args()
    if args.lstm:
        model = LSTM(1)
    else:
        model = MLP(1)
    train_commodities, train_stock, test_commodities, test_stock = get_data(constants.stock_filepath, constants.commodities_filepaths, constants.start_date_train, constants.end_date_train, constants.start_date_test, constants.end_date_test)
    accuracy = []
    test_loss = []
    for i in range(1, constants.EPOCH + 1):
        loss = train(model, train_commodities, train_stock)
        print(f"Train Loss for EPOCH {i}: {loss}")
        test_loss_per_epoch, accuracy_per_epoch = test(model, test_commodities, test_stock)
        accuracy.append(accuracy_per_epoch)
        test_loss.append(test_loss_per_epoch)
        print(f"Test Loss for EPOCH {i}: {sum(test_loss) / i}")
        print(f"Test Accuracy for EPOCH {i}: {sum(accuracy) / i}")
    viz_loss(test_loss)
    viz_accuracy(accuracy)
    print(f"Accuracy after {constants.EPOCH} epochs: {sum(accuracy) / constants.EPOCH}")
    print(model.summary())

if __name__ == "__main__":
    main()
