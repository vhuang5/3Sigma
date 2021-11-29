import argparse
from preprocessing import get_data
import tensorflow as tf
import constants
import lstm
import mlp
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
    total_loss = []
    for i in range(0,len(train_stock), model.batch_size):
        batched_commodities = train_commodities[i:i+model.batch_size]
        batched_stock = train_stock[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            pred = model(batched_commodities)
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
    total_loss = []
    total_accuracy = []
    for i in range(0,len(test_stock), model.batch_size):
        batched_commodities = test_commodities[i:i+model.batch_size]
        batched_stock = test_stock[i:i+model.batch_size]
        pred = model(batched_commodities)
        loss = model.loss(pred, batched_stock)
        total_loss += loss
        acc = model.accuracy(pred, batched_stock)
        total_accuracy += acc
        # print(loss)
    return total_loss/model.batch_size, total_accuracy/model.batch_size

def main():
    args = parse_args()
    if args.lstm:
        model = lstm
    else:
        model = mlp
    train_stock, train_commodities, test_stock, test_commmodities = get_data(constants.stock_filepath, constants.commodities_filepaths, constants.start_date_train, constants.end_date_train, constants.start_date_test, constants.end_date_test)
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
