import torch
import torchvision
import os
import pickle
import sys
import types

from models.model import LeNet5

class ExperimentManager():
    def __init__(self, root_dir=None):
        self.root_dir = root_dir
        if root_dir is not None:
            self.experiment_dir = self.get_next_experiment_folder()
        self.loss = None
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        self.config_dict = None
        self.model = None


    def get_next_experiment_folder(self):
        i = 1
        while os.path.isdir(os.path.join(self.root_dir,f'Experiment_{i}')):
            i += 1

        experiment_path = os.path.join(self.root_dir, f'Experiment_{i}')
        os.mkdir(experiment_path)
        return experiment_path

    def save_model(self, model):
        model_save_path = os.path.join(self.experiment_dir, 'model.pth')
        torch.save(model.state_dict(), model_save_path)

    def save_stats(self, loss=None, train_acc=None, val_acc=None, test_acc=None):
        self.loss = loss
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc

        if loss is not None:
            loss_path = os.path.join(self.experiment_dir, 'loss.pkl')
            with open(loss_path, 'wb') as f:
                pickle.dump(loss, f)

        if train_acc is not None:
            train_acc_path = os.path.join(self.experiment_dir, 'training_accuracies.pkl')
            with open(train_acc_path, 'wb') as f:
                pickle.dump(train_acc, f)

        if val_acc is not None:
            val_acc_path = os.path.join(self.experiment_dir, 'validation_accuracies.pkl')
            with open(val_acc_path, 'wb') as f:
                pickle.dump(val_acc, f)

        if test_acc is not None:
            test_acc_path = os.path.join(self.experiment_dir, 'testing_accuracies.pkl')
            with open(test_acc_path, 'wb') as f:
                pickle.dump(test_acc, f)

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


    def load(self, root_dir, experiment_num):
        self.experiment_dir = os.path.join(root_dir, f'Experiment_{experiment_num}')
        exists = os.path.exists(self.experiment_dir)

        if not exists:
            print(f'Experiment directory path {self.experiment_dir} did not exist.')
            sys.exit()


        with open(os.path.join(self.experiment_dir, 'loss.pkl'), 'rb') as f:
            self.loss = pickle.load(f)
        with open(os.path.join(self.experiment_dir, 'training_accuracies.pkl'), 'rb') as f:
            self.train_acc = pickle.load(f)
        val_path = os.path.join(self.experiment_dir, 'validation_accuracies.pkl')
        if os.path.exists(val_path):
            with open(val_path, 'rb') as f:
                self.val_acc = pickle.load(f)
        with open(os.path.join(self.experiment_dir, 'testing_accuracies.pkl'), 'rb') as f:
            self.test_acc = pickle.load(f)
        with open(os.path.join(self.experiment_dir, 'config.pkl'), 'rb') as f:
            self.config_dict = pickle.load(f)

        self.model = LeNet5(self.config_dict['num_classes'])
        self.model.load_state_dict(torch.load(os.path.join(self.experiment_dir, 'model.pth')))
