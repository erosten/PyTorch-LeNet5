import numpy as np




# W is the width of the square input image
# K is the filter size (Can be height or width)
# P is the padding used on a single dimension,
#    -- if we add one 0 on each side of the image, P=1
# S is the stride size
# Padding is by default 0, and ceiling is assumed to be used
def conv_output_size(W,K,S,P=0):
  return np.ceil((W - K + 2*P) / S) + 1


# K is the filter size
# C is the number of input channels/feature maps
# M is the number of output channels/feature maps
# Add 1 for bias
def conv_layer_parameters(K, C, M):
    return (K * K * C + 1)* M


# C_in is the number of input nodes
# C_out is the number of output nodes
# add C_out for biases
def fc_layer_parameters(C_in, C_out):
    return C_in * C_out + C_out


# as in LeNet5, pooling layers have a weight and bias associated
# with the output of the average pooling. Thus, the number of parameters
# are a weight and bias (2) * the number of input feature maps.
def pooling_layer_parameters(C_in):
    return C_in * 2
