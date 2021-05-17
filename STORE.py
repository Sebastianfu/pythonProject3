
## The formula for calculating the number of 0 to be filled is:
if (input_tensor_inColumn % self.stride_shape_inColumn ) == 0:
    totalPadding_inColumn = max(self.convolution_shape_inColumn - self.stride_shape_inColumn, 0)
else:
    totalPadding_inColumn = max(self.convolution_shape_inColumn - (input_tensor_inColumn % self.stride_shape_inColumn), 0) ##The role of the number 0 is to take effect when the stride value is greater than input

if (input_tensor_inRow % self.stride_shape_inRow) == 0:
    totalPadding_inRow = max(self.convolution_shape_inRow - self.stride_shape_inRow, 0)
else:
    totalPadding_inRow = max(self.convolution_shape_inRow - (input_tensor_inRow % self.stride_shape_inRow), 0)

## After determining the total number to be filled, we need to determine the number of zeros filled in the four directions
padding_inTop = totalPadding_inColumn // 2
padding_inBottom = totalPadding_inColumn -padding_inTop
padding_inLeft = totalPadding_inRow // 2
padding_inRight = totalPadding_inRow - padding_inLeft

## Use the pad part in NUMPY for padding
padding_Output = np.pad(input_tensor, ((padding_inLeft, padding_inRight), (padding_inTop, padding_inBottom), (0, 0)),
                        'constant')

## Next, calculate the Total output size
## https://blog.csdn.net/rain6789/article/details/78754516
output_inRow = math.ceil(input_tensor_inRow / self.stride_shape_inRow)
output_inColumn = math.ceil(input_tensor_inColumn / self.stride_shape_inColumn)
output_inChannel = math.ceil(input_tensor_inChannel / self.convolution_shape_inChannel)

outputs = np.empty(shape=(output_inColumn, output_inRow, output_inChannel))
for channel, z in enumerate(range(0, padding_Output.shape[2] - self.convolution_shape_inChannel + 1)):
    for row, x in enumerate(
            range(0, padding_Output.shape[0] - self.convolution_shape_inRow + 1, self.stride_shape_inRow)):
        for column, y in enumerate(
                range(0, padding_Output.shape[1] - self.convolution_shape_inColumn + 1, self.stride_shape_inColumn)):
            outputs[row][column][channel] = np.sum(
                padding_Output[x: x + self.convolution_shape_inRow, y: y + self.convolution_shape_inColumn,
                z: z + self.convolution_shape_inChannel] * self.convolutionShape)

## TEST OUTPUTS
print("OUTPUTS IS " + str[outputs] + '\n')

return outputs
