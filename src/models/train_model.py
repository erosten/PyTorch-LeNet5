import torch
import numpy as np

import config
from model import LeNet5

import sys
import pickle

sys.path.append('..')
from utils.experiment_manager import ExperimentManager


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
    experiments_root_dir = config.experiments_root_dir
    ignored_config_vals = config.ignored_config_vals

except Exception:
    print('config.py file not found.')
    sys.exit()



def evaluate(data_set_key='train'):
    data_set = data_sets_dict[data_set_key][0]
    data_loader = data_sets_dict[data_set_key][1]

    num_correct = 0
    num_total = 0
    for iter, data in enumerate(data_loader):
        imgs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            predictions = model(imgs)
            predicted_classes = torch.argmax(predictions, dim=1)
            num_correct += (labels == predicted_classes).sum()

    return num_correct.item() / len(data_set)

def train():
    epoch_loss = 0.0
    data_loader = data_sets_dict['train'][1]
    for iter, data in enumerate(data_loader):
        # load data
        imgs, labels = data[0].to(device), data[1].to(device)
        # update network
        optimizer.zero_grad()
        predictions = model(imgs)
        loss = loss_func(predictions, labels)
        loss.backward()
        optimizer.step()

        #bookkeeping
        epoch_loss += loss.item()
    return epoch_loss


if __name__ == '__main__':



    ### Instantiate Model and Optimizer
    model = LeNet5(data_sets_dict['num_classes']).to(device)
    optimizer = optimizer_func(model.parameters(), **optimizer_args)

    ### Train!
    EM = ExperimentManager(experiments_root_dir)
    loss = []
    training_accuracy = []
    testing_accuracy = []
    best_training_accuracy = 0
    for epoch in range(num_epochs):
        epoch_loss = train()
        epoch_training_accuracy = evaluate('train')
        epoch_test_accuracy = evaluate('test')

        #bookkeeping
        loss.append(epoch_loss)
        training_accuracy.append(epoch_training_accuracy)
        testing_accuracy.append(epoch_test_accuracy)

        print(f'Epoch: {epoch}, Loss: {epoch_loss}, Training Acc: {epoch_training_accuracy}')
        print(f'----- Testing Acc: {epoch_test_accuracy}')

        # if we have a new best training accuracy, save the model
        if epoch_training_accuracy > best_training_accuracy:
            EM.save_model(model)
            best_training_accuracy = epoch_training_accuracy
            print(f'----- New best training accuracy: {best_training_accuracy}, Saving Model')

    ### Save Results
    EM.save_stats(loss=loss, train_acc=training_accuracy, test_acc=testing_accuracy)
    EM.save_config(config, ignored_config_vals)
    print(f'Results saved to {EM.experiment_dir}')
