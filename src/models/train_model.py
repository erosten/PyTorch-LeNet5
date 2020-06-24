import torch
import numpy as np

import config
from model import LeNet5

import sys
import pickle
import time

from utils.experiment_manager import ExperimentManager
from utils.definitions import EXPERIMENTS_ROOT_DIR

torch.manual_seed(42)

### Config options
try:
    data_sets_dict = config.data_sets_dict

    batch_size_train = config.batch_size_train
    batch_size_test = config.batch_size_test

    loss_func = config.loss_func
    optimizer_func = config.optimizer_func
    optimizer_args = config.optimizer_args

    device = config.device
    num_epochs = config.num_epochs
    ignored_config_vals = config.ignored_config_vals

except Exception:
    print('config.py file not found.')
    sys.exit()


# calculates the accuracy
def evaluate(data_set_key='train'):
    data_set = data_sets_dict[data_set_key][0]
    data_loader = data_sets_dict[data_set_key][1]

    num_correct = 0
    num_total = 0
    for iter, data in enumerate(data_loader):
        imgs, labels = data[0].to(device), data[1].to(device)

        # do not update model parameter gradients
        with torch.no_grad():
            predictions = model(imgs)
            predicted_classes = torch.argmax(predictions, dim=1)
            num_correct += (labels == predicted_classes).sum()

    return num_correct.item() / len(data_set)

# runs 1 epoch (full pass through the training set)
def train():
    epoch_loss = 0.0
    # load the training data loader for the config specified training set
    data_loader = data_sets_dict['train'][1]
    for iter, data in enumerate(data_loader):
        # load data to device specified by config.py
        imgs, labels = data[0].to(device), data[1].to(device)

        ## update network
        # zero gradients in all network parameters
        optimizer.zero_grad()
        # get the predicted label probabilities
        # this is of size [batch_size_train, num_labels]
        preds = model(imgs)

        # loss_func specified in config.
        # CrossEntropyLoss defined in config expects the mismatch
        # between preds and labels, which is of size [batch_size_train]
        loss = loss_func(preds, labels)

        # compute the gradient of the loss w.r.t to all graph leaves (inputs)
        loss.backward()

        # perform a parameter update using calculated gradients
        optimizer.step()

        #bookkeeping
        epoch_loss += loss.item()

    return epoch_loss


if __name__ == '__main__':



    ### Instantiate model before optimizer if moving to cuda
    model = LeNet5(data_sets_dict['num_classes']).to(device)
    optimizer = optimizer_func(model.parameters(), **optimizer_args)

    ### Train

    # create up here since we use it to save our models
    EM = ExperimentManager(EXPERIMENTS_ROOT_DIR)

    # bookkeeping
    loss = []
    training_accuracy = []
    testing_accuracy = []
    best_training_accuracy = 0

    # Train the model
    for epoch in range(num_epochs):

        # run one training epoch
        epoch_loss = train()
        #
        epoch_training_accuracy = evaluate('train')
        epoch_test_accuracy = evaluate('test')

        #bookkeeping
        loss.append(epoch_loss)
        training_accuracy.append(epoch_training_accuracy)
        testing_accuracy.append(epoch_test_accuracy)


        # if we have a new best training accuracy, save the model
        if epoch_training_accuracy > best_training_accuracy:
            EM.save_model(model)
            best_training_accuracy = epoch_training_accuracy


        progress_update = (
            f'Epoch:                {epoch+1}/{num_epochs}\n' \
            f'Loss:                 {epoch_loss:.4f}\n' \
            f'Training Acc:         {epoch_training_accuracy*100:.4f}\n' \
            f'Testing Acc:          {epoch_test_accuracy*100:.4f}\n' \
            f'Best Test Acc:        {best_training_accuracy*100:.4f}\n'
            f'Last Saved Model:     Epoch {training_accuracy.index(best_training_accuracy)+1}\n' \
        )
        print(progress_update)

    ### Save Results to an ExperimentManager Object
    EM.save_stats(loss=loss, train_acc=training_accuracy, test_acc=testing_accuracy)
    EM.save_config(config, ignored_config_vals)
    print(f'Results saved to {EM.experiment_dir}')
