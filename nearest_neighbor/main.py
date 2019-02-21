# Russell Reinart, Stat 760, Spring 2019

import matplotlib.pyplot as plt
from k_nearest_neighbor import *


def plot_histograms(hists, k):
    fig, axes = plt.subplots(nrows=5, ncols=2)
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()
    
    plt.setp(
        axes, xticks=np.linspace(0,9,10))

    idx = 0
    for ax in axes.flatten():

        ax.bar(np.linspace(0,9,10), hists[idx])
        ax.set_ylabel(str(idx))
        ax.grid()
        idx += 1
    plt.show()

    return


def main():
    knn_classifier = k_nearest_neighbor("zip.train", "zip.test")
    losses = []
    ks = []
    for k in range(1, 15, 2):
        # get predicted classes
        y_predict = knn_classifier.classify(k)

        # compute and plot loss statistics
        stats = knn_classifier.compute_loss_statistics(y_predict)
        plot_histograms(stats, k)        
        _, loss_fraction = knn_classifier.compute_loss(y_predict)
        ks.append(k)
        losses.append(loss_fraction)
        print('k = %i, loss fraction = %f' %(k, loss_fraction))

    # plot loss vs # of neighbors
    plt.figure()
    plt.xticks(np.linspace(0, 2*len(losses), 2*len(losses)+1))
    plt.plot(ks, losses)
    plt.title('misclassified fraction vs. # of neighbors')
    plt.grid()
    plt.show()

    return

if __name__ == "__main__":
    main()

