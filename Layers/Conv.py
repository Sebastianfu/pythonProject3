import numpy as np
import math
from scipy import ndimage

## Conv
class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.convKernels = num_kernels ## Convolution kernel
        ##Stride in ROW OR in COLUMN
        self.stride_shape_inRow = None
        self.stride_shape_inColumn = None
        self.convolutionShape = None
        self.weights = None
        self.bias =None
        self.filter = None
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._input_size = None


        ## return the gradient with respect to the weights and bias, after they have been calculated in the backward-pass.
        self.backwardWeights = None
        self.backwardBias = None
        ## Channel, Row and Column of the convolution_shape
        self.convolution_shape_inChannel = self.convolution_shape[0]
        self.convolution_shape_inM = self.convolution_shape[1]
        if convolution_shape[2] is not None: ## 1-D or 2-D
            self.convolution_shape_inN = convolution_shape[2]
            self.weights = np.random.randn(self.convolution_shape * self.convKernels)
        else:
            self.weights = np.random.randn(self.convolution_shape + self.convKernels)



        ##Determine the type of parameter, is it string or tuple
        if len(self.stride_shape) > 1:
            self.stride_shape_inRow = self.stride_shape[0]
            self.stride_shape_inColumn = self.stride_shape[1]
            ## Test Row & Column
            print("stride_shape is "+ str(self.stride_shape)+ "\n")
            print("stride_shape_inRow is "+ str(self.stride_shape_inRow)+ "\n")
            print("stride_shape_inColumn is " + str(self.stride_shape_inColumn) + "\n")
        else:
            self.stride_shape_1D_length = int(self.stride_shape)
            print("stride_shape_1D_length is " + str(self.stride_shape_1D_length) + "\n")

        ##Initialize variables convolutionShape random in the range [0, 1).

        self.convolutionShape = np.random.random(self.convolution_shape) #https://blog.csdn.net/qq_38701868/article/details/99226311?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control
        ##Test convolutionShape
        print("convolutionShape is " + str(self.convolutionShape) + "\n")



    ## provide two properties:gradient weights and gradient bias, which return the gradient with respect to the weights and bias, after they have been calculated in the backward-pass.
    ## setter grandient weights
    def gradient_weights(self,granient_weight):
        self._gradient_weights = granient_weight

    ## getter gradient_weights
    def radient_weights(self):
        return self._gradient_weights

    ## setter grandient bias
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    ## getter grandient bias
    def gradient_bias(self):
        return self._gradient_bias



    def forward(self,input_tensor):
        input_tensor_inTotal = np.shape(input_tensor) ## input_tensorTotal represents the size of the input
        self._input_size = input_tensor_inTotal

        ## if input is a 1-D array
        if input_tensor_inTotal[3] is None:
            output_Correlate = ndimage.correlate1d(input_tensor, self.filter, 'constant',cval=0)
            output_Final = output_Correlate + self.bias
            return output_Final
        ## if input is 2-D , 3-D or more
        else:
            output_Correlate = ndimage.correlate(input_tensor, self.filter, 'constant', cval=0)
            output_Final = output_Correlate + self.bias
            return output_Final



    ## setter Optimizer
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    ## getter optimizer
    def optimizer(self):
        return self._optimizer

    def backward(self, error_tensor):
        error_tensor_inTotal = np.shape(error_tensor)
        ## if input is a 1-D array
        if error_tensor_inTotal[3] is None:
            output_Convolve = ndimage.correlate1d(error_tensor, self.filter,'constant',cval=0)
            if self._optimizer is not None:
                self

        return

