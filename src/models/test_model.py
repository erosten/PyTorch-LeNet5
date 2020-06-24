import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import sys
import os

from model import LeNet5
import config


from utils.experiment_manager import ExperimentManager
from utils.definitions import EXPERIMENTS_ROOT_DIR, DATA_DIR

torch.manual_seed(42)

### Config options
try:
    data_sets_dict = config.data_sets_dict
    batch_size_test = config.batch_size_test
    transform = config.transform

    device = config.device

except Exception:
    print('config.py file not found.')
    sys.exit()

# similar to train_model.evaluate
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

## Uses Experiment Manager to load model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-en','--experiment_num', help='Experiment Number you\'d like to test', required=True)
    args = parser.parse_args()

    experiment_num = args.experiment_num

    EM = ExperimentManager()
    EM.load(EXPERIMENTS_ROOT_DIR, experiment_num)
    # device specified in config
    model = EM.model.to(device)
    # test accuracy on MNIST test set
    test_accuracy(model, data_set=data_sets_dict['test'][0], data_loader=data_sets_dict['test'][1])


    # test on your own images
    # play with code below as needed

    # define a rotation transform since my images are strangely flipped
    class rotationTransform:
        def __init__(self, angle):
            self.angle = angle

        def __call__(self,x):
            return torchvision.transforms.functional.rotate(x,self.angle)

    # apply thresholding to create
    # 1. Black background: values larger than mean_val - coeff*std_val set to 0
    # 2. White background: all other values set to 1
    class thresholdTransform:
        def __init__(self, coeff):
            self.coeff = coeff
        def __call__(self,x):
            # convert PIL Image to array
            arr = np.array(x)
            arr[arr>(np.mean(arr) - self.coeff*np.std(arr))] = 0
            arr[arr > 255] = 255
            # convert back to PIL Image
            x = Image.fromarray(np.uint8(arr), 'L')
            return x


    # Compose transforms.
    # 1. CenterCrop as images are massive - 4032,3024
    # 2. Convert to Grayscale
    # 3. Resize to 32x32 with interpolation LANCZOS
    # 3a. interpolation number to PIL interpolation: https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Resize
    # 3b. LANCZOS best interpolation for downscaling: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
    # 4. Rotate the Image
    # 5. Threshold the Image
    # 6. Convert to tensor (which normalizes to [0,1])
    transform = transforms.Compose(
        [transforms.CenterCrop(1500),
         transforms.Grayscale(),
         transforms.Resize(32,interpolation=Image.LANCZOS),
         rotationTransform(-90),
         thresholdTransform(coeff=1),
         transforms.ToTensor()
        ])

    # Load the custom dataset
    custom_data_image_folder = os.path.join(DATA_DIR, 'my_data')

    data = torchvision.datasets.ImageFolder(root=custom_data_image_folder, transform=transform)
    data_loader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)


    # get the images, and reshape them to 28x28
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.squeeze(1)


    # Compute test accuracy and plot the images
    _, predictions, labels = test_accuracy(model, data_set=data, data_loader=data_loader)


    plt.figure(1)
    for i in range(len(labels)):
        plt.subplot(2,5,i+1)
        plt.imshow(images[i].numpy(), cmap='gray')
        plt.title(f'Prediction: {predictions[i]}, Label: {labels[i]}', fontsize=7)
    plt.show()
