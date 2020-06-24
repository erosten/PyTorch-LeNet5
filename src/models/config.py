import torch
import torchvision
import torchvision.transforms as transforms
from definitions import DATA_DIR

# gpu if exists, cpu else
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# batch size to use on the training set
batch_size_train = 256

# batch size to use on the testing set
batch_size_test = 1024



# number of epochs to train on
num_epochs = 100

# loss function to use
loss_func = torch.nn.CrossEntropyLoss()

# optimizer_func = torch.optim.SGD
# learning_rate = 1
# momentum = 0.9
# nesterov = True
# define all optimizer parameters in a dictionary to pass at training time
# optimizer_args = {'lr': learning_rate, 'momentum': momentum}
# optimizer_args = {'lr': learning_rate, 'momentum': momentum, 'nesterov': nesterov}


optimizer_func = torch.optim.Adam
learning_rate = 0.001
optimizer_args = {'lr': learning_rate}





###  Data options
dataset = 'MNIST'

if dataset == 'MNIST':
    transform = transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor()
        ])


    training_set = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set,
                    batch_size=batch_size_train, shuffle=True, num_workers=8)

    test_set = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                    batch_size=batch_size_test, shuffle=False, num_workers=8)

    num_classes = len(training_set.classes)

    data_sets_dict = {'train': (training_set, train_loader),
                 'test' : (test_set, test_loader),
                 'num_classes': num_classes}

    ignored_config_vals = [training_set, train_loader, test_set, test_loader, data_sets_dict]

else:
    print(f'Dataset {dataset} not recognized')
