from matplotlib import pyplot as plt

def viz_accuracy(acc):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(acc))]
    plt.plot(x, acc)
    plt.title('Test Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()  

def viz_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()  

def viz_predictions(pred, labels):
    """
    Uses Matplotlib to visualize the our predicted SPY prices vs the actual SPY prices
    :param pred: a 1-D array of our predicted prices
    :param labels: a 1-D array of actual SPY prices
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(pred))]
    plt.plot(x, pred, label='Predicted')
    plt.plot(x, labels, label='Actual')
    plt.legend()
    plt.title('S&P 500 Price Prediction')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.show()  

