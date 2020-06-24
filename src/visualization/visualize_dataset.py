import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size_train = 32

### Load Data
transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor()
    ])

training_set = torchvision.datasets.MNIST('../../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training_set,
                batch_size=batch_size_train, shuffle=True, num_workers=8)

def imshow(img):
    npimg = img.numpy()
    # pytorch images typically in form (batch_size, input_channels, height, width)
    # since here our batch_size is nonexistent, we want images in form
    # height, width, channels
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title('Random MNIST Training Set Images')
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next() # load batch_size_train number of images/labels
imshow(torchvision.utils.make_grid(images))
