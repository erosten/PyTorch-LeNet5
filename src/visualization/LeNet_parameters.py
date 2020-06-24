

import sys
sys.path.append('..')

from utils.parameters import conv_output_size
from utils.parameters import conv_layer_parameters
from utils.parameters import fc_layer_parameters
from utils.parameters import pooling_layer_parameters

from tabulate import tabulate

headers=['Layer', 'Output Size', 'Trainable Parameters']
input = ['Input', '32 x 32', '']

# Layer 1 - Conv
W = 32
K = 5
S = 1
out1 = conv_output_size(W,K,S)
C_in = 1
C_out = 6
params1 = conv_layer_parameters(K,C_in,C_out)
layer1 = ['Conv1', f'{str(int(out1))} x {str(int(out1))}', params1]


# Layer 1 - Pooling
W = out1
K = 2
S = 2
out1_pool = conv_output_size(W,K,S)
params1_pool = pooling_layer_parameters(C_out)
layer1pool = ['Pool1', f'{str(int(out1_pool))} x {str(int(out1_pool))}', params1_pool]

# Layer 2 - Conv - Special connection scheme for LeNet given in notes & confirms paper results
W = out1_pool
K = 5
S = 1
out2 = conv_output_size(W,K,S)
## 6 layers with 3 input feature maps
C_in = 3
C_out = 6
params_2_1 = conv_layer_parameters(K,C_in,C_out)
## 9 layers with 4 input feature maps
C_in = 4
C_out = 9
params_2_2 = conv_layer_parameters(K,C_in,C_out)
## 1 layer with 6 input feature maps
C_in = 6
C_out = 1
params_2_3 = conv_layer_parameters(K,C_in,C_out)

## sum results
C_out = 6 + 9 + 1
params2 = params_2_1 + params_2_2 + params_2_3
layer2 = ['Conv2', f'{str(int(out2))} x {str(int(out2))}', params2]



#  Layer 2 - Pooling
W=out2
K = 2
S = 2
out2_pool = conv_output_size(W,K,S)
params2_pool = pooling_layer_parameters(C_out)
layer2pool = ['Pool2', f'{str(int(out2_pool))} x {str(int(out2_pool))}', params2_pool]

# Layer 3 - Conv
W = out2_pool
K = 5
S = 1
out3 = conv_output_size(W,K,S)
C_in = C_out
C_out = 120
params3 = conv_layer_parameters(K,C_in,C_out)
layer3 = ['Conv2', C_out, params3]


# Layer 4 - FC1
C_in = C_out * out3 * out3 # flattened num of feature maps * kernel_size^2
C_out = 84
params4 = fc_layer_parameters(C_in, C_out)
layer4 = ['FC1', C_out, params4]

# Layer 5 - RBF
C_in = C_out
C_out = 10 # number of classes for MNIST
params5 = 0
layer5 = ['RBF', C_out, params5]





fc_params = params4
conv_pool_params = params1+params1_pool+params2+params2_pool+params3
total_params = fc_params + conv_pool_params

total = ['Total', '', total_params]

print('\n')
print(tabulate([input,layer1, layer1pool, layer2, layer2pool, layer3, layer4, layer5, total],
        headers=headers, tablefmt='orgtbl'))

print('\n')
