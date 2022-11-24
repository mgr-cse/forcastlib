from matplotlib import pyplot as plt

def plot_graph(Y_train, Y_test, pred_mean, pred_mean_low, pred_mean_high):
    plt.figure(figsize=(8,5))
    x_pred = range(len(Y_train)+1, len(pred_mean)+len(Y_train)+1)
    plt.plot(Y_train)
    plt.plot(x_pred, pred_mean)
    plt.plot(x_pred, Y_test, 'o', markersize=1)
    plt.fill_between(x_pred, pred_mean+pred_mean_low, pred_mean+pred_mean_high, alpha = 0.3, color = 'orange')
    plt.legend(['Train', 'Predicted', 'Actual', 'Deviation'])
    plt.xlabel('Time')
    plt.show()