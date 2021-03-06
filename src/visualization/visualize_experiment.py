import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from utils.experiment_manager import ExperimentManager
from utils.definitions import EXPERIMENTS_ROOT_DIR


def main(experiment_num):
    # try to load the experiment manager
    EM = ExperimentManager()
    EM.load(EXPERIMENTS_ROOT_DIR, experiment_num)

    # calculate epochs from loss list
    epochs = list(range(len(EM.loss)))
    epochs = [epoch + 1 for epoch in epochs]

    # Report the maximum test accuracy over all epochs
    max_test_acc = max(EM.test_acc)
    max_test_acc_epoch = epochs[EM.test_acc.index(max_test_acc)]
    print(f'Maximum Test Accuracy was {max_test_acc} on epoch {max_test_acc_epoch}')

    # Report the maximum train accuracy over all Epochs
    # and corresponding test accuracy (can be less than max)
    max_train_acc = max(EM.train_acc)
    max_train_acc_epoch = epochs[EM.train_acc.index(max_train_acc)]
    print(f'Maximum Training Accuracy was {max_train_acc} on epoch {max_train_acc_epoch}')

    print(f'---Corresponding Test Accuracy: {EM.test_acc[EM.train_acc.index(max_train_acc)]}')

    # Plot
    plt.figure()
    plt.plot(epochs, EM.loss, linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')


    plt.figure()
    plt.plot(epochs, EM.train_acc, linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Epochs vs Training Accuracy')


    plt.figure()
    plt.plot(epochs, EM.test_acc, linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.title('Epochs vs Testing Accuracy')

    plt.show()

## experiment number is the number used to load an experiment manager
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en','--experiment_num', help='Experiment Number you\'d like to visualize', required=True)
    args = parser.parse_args()

    experiment_num = args.experiment_num

    print(f'Visualizing Experiment {experiment_num} in {EXPERIMENTS_ROOT_DIR}')
    main(experiment_num)
