import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

from model import LeNet5
import config

import sys
sys.path.append('..')
from utils.experiment_manager import ExperimentManager

torch.manual_seed(42)

### Config options
try:
    data_sets_dict = config.data_sets_dict
    batch_size_test = config.batch_size_test
    transform = config.transform

    device = config.device
    experiments_root_dir = config.experiments_root_dir

except Exception:
    print('config.py file not found.')
    sys.exit()

def test_accuracy(model, data_set, data_loader):
    num_correct = 0
    num_total = 0
    for iter, data in enumerate(data_loader):
        imgs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            predictions = model(imgs)
            predicted_classes = torch.argmax(predictions, dim=1)
            num_correct += (labels == predicted_classes).sum()

    test_accuracy = num_correct.item() / len(data_set)

    print(f'Test Accuracy: {test_accuracy}')
    return test_accuracy, predicted_classes, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en','--experiment_num', help='Experiment Number you\'d like to visualize', required=True)
    args = parser.parse_args()

    experiment_num = args.experiment_num

    EM = ExperimentManager()
    EM.load(experiments_root_dir, experiment_num)

    model = EM.model.to(device)
    # test accuracy on MNIST test set
    test_accuracy(model, data_set=data_sets_dict['test'][0], data_loader=data_sets_dict['test'][1])


    # test on your own images
    # play with this as needed
    class rotationTransform:
        def __init__(self, angle):
            self.angle = angle

        def __call__(self,x):
            return torchvision.transforms.functional.rotate(x,self.angle)

    class normalizeTransform:
        def __init__(self, coeff):
            self.coeff = coeff
        def __call__(self,x):

            arr = np.array(x)
            arr[arr>(np.mean(arr) - self.coeff*np.std(arr))] = 0
            arr[arr > 255] = 255
            x = Image.fromarray(arr)
            # bbox = x.getbbox()
            # print(bbox)
            # cropped = x.crop(bbox)
            # x.show()
            return x

    class invertTransform:
        def __init__(self):
            pass
        def __call__(self,x):
            return PIL.ImageOps.invert(x)

    rotation_transform = rotationTransform(-90)
    coeff = 1 # for my_data
    coeff = 0 # for my_data2
    normalize_transform = normalizeTransform(coeff)
    invert_transform = invertTransform()

    transform = transforms.Compose(
        [transforms.CenterCrop(1500),
         transforms.Grayscale(),
         # normalize_transform,
         transforms.Resize(32,interpolation=3),
         rotation_transform,
         normalize_transform,
         # invert_transform,
         transforms.ToTensor()
        ])

    custom_data_image_folder = '../../data/my_data2'

    data = torchvision.datasets.ImageFolder(root=custom_data_image_folder, transform=transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)


    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.squeeze(1)







    _, predictions, labels = test_accuracy(model, data_set=data, data_loader=data_loader)

    print(predictions)
    print(labels)

    plt.figure(1)
    for i in range(len(labels)):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title(f'Prediction: {predictions[i]}, Label: {labels[i]}', fontsize=7)
    plt.show()
