import torch
import torchvision
import os
import pickle
import sys
import types

from models.model import LeNet5

class ExperimentManager():
    def __init__(self, root_dir=None):
        # Directory where root_dir/Experiment_{i}
        self.root_dir = root_dir
        # If we are loading, do not pass root_dir
        if root_dir is not None:
            self.experiment_dir = self._get_next_experiment_folder()

        # experiment variables of interest
        self.loss = None
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        self.config_dict = None
        self.model = None


    def _get_next_experiment_folder(self):
        i = 1
        while os.path.isdir(os.path.join(self.root_dir,f'Experiment_{i}')):
            i += 1

        experiment_path = os.path.join(self.root_dir, f'Experiment_{i}')
        os.mkdir(experiment_path)
        return experiment_path

    # save a pytorch model
    def save_model(self, model):
        model_save_path = os.path.join(self.experiment_dir, 'model.pth')
        torch.save(model.state_dict(), model_save_path)

    ## stat assumed as list with len(stat) = num_epochs
    def save_stat(self, stat, stat_filename):
        stat_path = os.path.join(self.experiment_dir, stat_filename)
        with open(stat_path, 'wb') as f:
            pickle.dump(stat, f)


    # save training stats: loss, train_acc, val_acc, test_acc
    def save_stats(self, loss=None, train_acc=None, val_acc=None, test_acc=None):
        self.loss = loss
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc

        if loss is not None:
            self.save_stat(loss, 'loss.pkl')

        if train_acc is not None:
            self.save_stat(train_acc, 'training_accuracies.pkl')

        if val_acc is not None:
            self.save_stat(val_acc, 'validation_accuracies.pkl')

        if test_acc is not None:
            self.save_stat(test_acc, 'testing_accuracies.pkl')


    # save config file
    # saves all variables that are:
    # 1. not modules
    # 2. not in the passed ignored_vals list
    # 3. not the ignored_vals_list itself
    def save_config(self, config, ignored_vals):
        hyperparameters = [item for item in dir(config) if not item.startswith('__')]
        config_dict = {}
        variables = vars(config)
        for hyperparam in hyperparameters:
            key = hyperparam
            value = variables[key]
            if not isinstance(value, types.ModuleType) and value not in ignored_vals and key != 'ignored_config_vals':
                config_dict[key] = value

        self.config_dict = config_dict
        config_path = os.path.join(self.experiment_dir, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config_dict, f)

    # attempts to load a state by given filename
    # if it does not exist, return an empty list
    def load_stat(self, stat_filename):
        stat_path = os.path.join(self.experiment_dir, stat_filename)
        if os.path.exists(stat_path):
            with open(stat_path,'rb') as f:
                return pickle.load(f)
        else:
            return []

    # loads an experiment given the experiment number and root experiment dir
    def load(self, root_dir, experiment_num):
        # set experiment dir and check if it exists
        self.experiment_dir = os.path.join(root_dir, f'Experiment_{experiment_num}')
        exists = os.path.exists(self.experiment_dir)

        if not exists:
            print(f'Experiment directory path {self.experiment_dir} did not exist.')
            sys.exit()

        # load stats
        self.loss = self.load_stat('loss.pkl')
        self.train_acc = self.load_stat('training_accuracies.pkl')
        self.val_acc = self.load_stat('validation_accuracies.pkl')
        self.test_acc = self.load_stat('testing_accuracies.pkl')
        self.config_dict = self.load_stat('config.pkl')

        # load model
        self.model = LeNet5(self.config_dict['num_classes'])
        self.model.load_state_dict(torch.load(os.path.join(self.experiment_dir, 'model.pth')))
