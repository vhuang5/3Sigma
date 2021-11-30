import matplotlib.pyplot as plt

def viz_accuracy(acc):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(acc))]
    plt.plot(x, acc)
    plt.title('Test Accuracy per Epoch LSTM')
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
    plt.title('Test Loss per Epoch LSTM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()  

