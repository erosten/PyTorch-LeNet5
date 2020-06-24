import pickle
import sys
import matplotlib.pyplot as plt
import argparse
sys.path.append('..')
from utils.experiment_manager import ExperimentManager

def main(experiment_num, experiments_root_dir):
    EM = ExperimentManager()
    EM.load(experiments_root_dir, experiment_num)

    epochs = list(range(len(EM.loss)))
    epochs = [epoch + 1 for epoch in epochs]

    max_test_acc = max(EM.test_acc)
    max_test_acc_epoch = epochs[EM.test_acc.index(max_test_acc)]
    print(f'Maximum Test Accuracy was {max_test_acc} on epoch {max_test_acc_epoch}')

    max_train_acc = max(EM.train_acc)
    max_train_acc_epoch = epochs[EM.train_acc.index(max_train_acc)]
    print(f'Maximum Training Accuracy was {max_train_acc} on epoch {max_train_acc_epoch}')

    print(f'---Corresponding Test Accuracy: {EM.test_acc[EM.train_acc.index(max_train_acc)]}')


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en','--experiment_num', help='Experiment Number you\'d like to visualize', required=True)
    parser.add_argument('--root_dir', help='Root Directory of Experiments in form root_dir\\Experiment_{en}')
    args = parser.parse_args()
    if args.root_dir is None:
        experiments_root_dir = '../../models'
    else:
        experiments_root_dir = args.root_dir

    experiment_num = args.experiment_num

    print(f'Visualizing Experiment {experiment_num} in {experiments_root_dir}')
    main(experiment_num, experiments_root_dir)
