# LeNet5

This repo implements a modified version of LeNet-5 in PyTorch as introduced in ["Gradient-Based Learning Applied to Document Recognition" by LeCun et al., 1998a](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf). Reports contains some of my own notes while reading the paper, the differences between this implementation and the original, and results on both the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), and my own handwritten digits.

![](https://github.com/erosten/PyTorch-LeNet5/blob/master/reports/images/my_results.png?raw=true)

## Running the code


First, clone the repo

```
git clone https://github.com/erosten/PyTorch-LeNet5.git
```

Change into the cloned repo directory
```
cd PyTorch-LeNet5/
```

Install the requirements (I strongly recommend a virtual environment)

```
pip3 install -r requirements.txt
```

Install the modules in the repo

```
pip3 install -e .
```

Start the training

```
python3 src/models/train_model.py
```


## File Information

The general file tree is shown below with some omissions for clarity.

```
.
├── data
│   ├── MNIST
│   ├── my_data
├── models
│   └── Experiment_6
│       ├── config.pkl
│       ├── loss.pkl
│       ├── model.pth
│       ├── testing_accuracies.pkl
│       └── training_accuracies.pkl
├── README.md
├── reports
│   ├── images
│   ├── notes.pdf
│   └── notes.tex
├── requirements.txt
└── src
    ├── models
    │   ├── config.py
    │   ├── model.py
    │   ├── test_model.py
    │   └── train_model.py
    ├── utils
    │   ├── definitions.py
    │   ├── experiment_manager.py
    │   ├── parameters.py
    └── visualization
        ├── LeNet_parameters.py
        ├── visualize_dataset.py
        └── visualize_experiment.py
```

### ./models

This directory stores experimental results such as model checkpoints, training and testing accuracies, and loss values with respect to epoch as pickle files. These experiment directories are automatically generated by `src/utils/experiment_manager.py` each time `train_model.py` is run. The best experiment (as can be found in the report) is given with trained weights, achieving ~99% accuracy on the MNIST test set.

### ./src/models

This directory contains 4 files

- `config.py`: Contains all the configurations necessary to run `train_model.py` such as the loss function, optimizer, dataset and batch sizes. This is where you should modify training hyperparameters
- `model.py`: Contains the LeNet5 model implementation
- `train_model.py`: Contains training code, and stores the data in an ExperimentManager object
- `test_model.py`: Contains code for testing a trained model on the MNIST dataset as well as my own (and your own) handwritten digits.

`test_model.py` requires an experiment to load the model from. To use the provided trained model from experiment 6, one can run

```
python test_model.py -en 6
```

### ./src/utils
This directory contains 3 files

- `definitions.py`: Contains definitions for different useful directories that are used throughout the rest of the src code
- `experiment_manager.py`: Contains the experiment manager object source code
- `parameters.py`: Contains useful calculation functions for trainable parameters and output sizes for convolutional and fully connected layers.

### ./src/visualization

`LeNet_parameters.py` uses these functions to calculate the number of trainable parameters as in the original paper. You can run this file directly, or simply refer to the output of it shown below

```
| Layer   | Output Size   | Trainable Parameters   |
|---------+---------------+------------------------|
| Input   | 32 x 32       |                        |
| Conv1   | 28 x 28       | 156                    |
| Pool1   | 14 x 14       | 12                     |
| Conv2   | 10 x 10       | 1516                   |
| Pool2   | 5 x 5         | 32                     |
| Conv2   | 120           | 48120                  |
| FC1     | 84            | 10164.0                |
| RBF     | 10            | 0                      |
| Total   |               | 60000.0                |
```

`visualize_dataset.py` loads the MNIST dataset and shows some example images

`visualize_experiment.py` loads an experiment into an ExperimentManager object, and provides accuracy and loss plots, as well as the best training and testing accuracies achieved during a training experiment. As with the `test_model.py` file, an experiment is required to use this file.

```
python visualize_experiment.py -en 6
```
