# Node Reference
## Display data
| Node | Description |
| --- | --- |
| [Pt Show Size](pt_show_size.md) | Displays PyTorch Size object as a string. |
| [Pt Show Text](pt_show_text.md) | Displays PyTorch tensor as a string. Note that the tensor is partially printed out when |
## IO
| Node | Description |
| --- | --- |
| [Pt Save Model](pt_save_model.md) | A wrapper class for saving a PyTorch model. |
## PyTorch wrapper - Arithmetic operations
| Node | Description |
| --- | --- |
| [Pt Add](pt_add.md) | Adds two PyTorch tensors. |
| [Pt Div](pt_div.md) | Divides one PyTorch tensor by another element-wise. |
| [Pt Floor Div](pt_floor_divide.md) | Performs element-wise floor division on two PyTorch tensors. |
| [Pt Mul](pt_mul.md) | Multiplies two PyTorch tensors element-wise. |
| [Pt Pow](pt_pow.md) | Raises one PyTorch tensor to the power of another element-wise. |
| [Pt Remainder](pt_remainder.md) | Computes the element-wise remainder of division between two PyTorch tensors. |
| [Pt Sub](pt_sub.md) | Subtracts one PyTorch tensor from another. |
## PyTorch wrapper - Bitwise operations
| Node | Description |
| --- | --- |
| [Pt Bitwise And](pt_bitwise_and.md) | Performs a bitwise AND operation on two PyTorch tensors element-wise. |
| [Pt Bitwise Left Shift](pt_bitwise_left_shift.md) | Performs a bitwise left shift operation on two PyTorch tensors element-wise. |
| [Pt Bitwise Not](pt_bitwise_not.md) | Performs a bitwise NOT operation on a PyTorch tensor element-wise. |
| [Pt Bitwise Or](pt_bitwise_or.md) | Performs a bitwise OR operation on two PyTorch tensors element-wise. |
| [Pt Bitwise Right Shift](pt_bitwise_right_shift.md) | Performs a bitwise right shift operation on two PyTorch tensors element-wise. |
| [Pt Bitwise Xor](pt_bitwise_xor.md) | Performs a bitwise XOR operation on two PyTorch tensors element-wise. |
## PyTorch wrapper - Comparison operations
| Node | Description |
| --- | --- |
| [Pt Eq](pt_eq.md) | Tests whether two PyTorch tensors are equal element-wise. |
| [Pt Ge](pt_ge.md) | Tests whether elements in the first PyTorch tensor are greater than or equal to the corresponding elements in the second tensor. |
| [Pt Gt](pt_gt.md) | Tests whether elements in the first PyTorch tensor are greater than the corresponding elements in the second tensor. |
| [Pt Le](pt_le.md) | Tests whether elements in the first PyTorch tensor are less than or equal to the corresponding elements in the second tensor. |
| [Pt Lt](pt_lt.md) | Tests whether elements in the first PyTorch tensor are less than the corresponding elements in the second tensor. |
| [Pt Ne](pt_ne.md) | Tests whether two PyTorch tensors are not equal element-wise. |
## PyTorch wrapper - Image processing
| Node | Description |
| --- | --- |
| [Pt Crop](pt_crop.md) | Crops a PyTorch tensor to the specified size. The input tensor must have a shape of (c, h, w) or (b, c, h, w). |
| [Pt From Image](pt_from_image.md) | Casts an Image tensor as a PyTorch tensor. |
| [Pt From Image Transpose](pt_from_image_transpose.md) | Casts an image tensor to a PyTorch tensor and transposes it from (H, W, C) to (C, H, W). For rank-4 inputs, the batch axis remains unchanged. |
| [Pt Interpolate By Scale Factor](pt_interpolate_by_scale_factor.md) | Resizes a PyTorch tensor using interpolation by scale factor. The input tensor must have a shape of (c, h, w) or (b, c, h, w). |
| [Pt Interpolate To Size](pt_interpolate_to_size.md) | Resizes a PyTorch tensor using interpolation. The input tensor must have a shape of (c, h, w) or (b, c, h, w). |
| [Pt Pad](pt_pad.md) | Pads a PyTorch tensor to the specified size. Padded area will be black. The input tensor must have a shape of (c, h, w) or (b, c, h, w). |
| [Pt To Image](pt_to_image.md) | Casts a PyTorch tensor as an Image tensor. |
| [Pt To Image Transpose](pt_to_image_transpose.md) | Casts a PyTorch tensor as an Image tensor and transposes it from (C, H, W) to (H, W, C). For rank-4 inputs, the batch axis remains unchanged. |
## PyTorch wrapper - Indexing and Slicing Operations
| Node | Description |
| --- | --- |
| [Pt Gather](pt_gather.md) | Generates a tensor based on the index tensor using PyTorch's `gather` function. |
| [Pt Index Select](pt_index_select.md) | Extracts elements from the input tensor along a specified dimension using an index tensor. |
| [Pt Masked Select](pt_masked_select.md) | Extracts elements from the input tensor whose corresponding value in `masked_tens` is `True`. |
| [Pt Scatter](pt_scatter.md) | Generates a new tensor by replacing values at specified positions using an index tensor. |
| [Pt Where](pt_where.md) | Generates a new tensor by selecting values based on a condition tensor. |
## PyTorch wrapper - Logical operations
| Node | Description |
| --- | --- |
| [Pt Logical And](pt_logical_and.md) | Performs a logical AND operation on two PyTorch tensors element-wise. |
| [Pt Logical Not](pt_logical_not.md) | Performs a logical NOT operation on a PyTorch tensor element-wise. |
| [Pt Logical Or](pt_logical_or.md) | Performs a logical OR operation on two PyTorch tensors element-wise. |
| [Pt Logical Xor](pt_logical_xor.md) | Performs a logical XOR operation on two PyTorch tensors element-wise. |
## PyTorch wrapper - Math operations
| Node | Description |
| --- | --- |
| [Pt Abs](pt_abs.md) | Computes the absolute value of each element in a PyTorch tensor. |
| [Pt Acos](pt_acos.md) | Computes the arccosine (inverse cosine) of a PyTorch tensor element-wise. |
| [Pt Asin](pt_asin.md) | Computes the arcsine (inverse sine) of a PyTorch tensor element-wise. |
| [Pt Atan](pt_atan.md) | Computes the arc tangent (inverse tangent) of a PyTorch tensor element-wise. |
| [Pt Cos](pt_cos.md) | Computes the cosine of a PyTorch tensor element-wise. |
| [Pt Cosh](pt_cosh.md) | Computes the hyperbolic cosine of a PyTorch tensor element-wise. |
| [Pt Exp](pt_exp.md) | Performs an exponential operation on a PyTorch tensor element-wise. |
| [Pt Log](pt_log.md) | Computes the natural logarithm (log base e) of a PyTorch tensor element-wise. |
| [Pt Neg](pt_neg.md) | Computes the negation of each element in a PyTorch tensor. |
| [Pt Sin](pt_sin.md) | Computes the sine of a PyTorch tensor element-wise. |
| [Pt Sinh](pt_sinh.md) | Computes the hyperbolic sine of a PyTorch tensor element-wise. |
| [Pt Sqrt](pt_sqrt.md) | Computes the square root of each element in a PyTorch tensor. |
| [Pt Tan](pt_tan.md) | Computes the tangent of a PyTorch tensor element-wise. |
| [Pt Tanh](pt_tanh.md) | Computes the hyperbolic tangent of a PyTorch tensor element-wise. |
## PyTorch wrapper - Matrix operations
| Node | Description |
| --- | --- |
| [Pt Bmm](pt_bmm.md) | Performs batched matrix multiplication on two 3D PyTorch tensors. |
| [Pt Einsum](pt_einsum.md) | Performs Tensor operations specified in the Einstein summation equation. |
| [Pt Mat Mul](pt_matmul.md) | Performs matrix multiplication on two PyTorch tensors. |
| [Pt Mm](pt_mm.md) | Performs 2D matrix multiplication on two PyTorch tensors. |
## PyTorch wrapper - Reduction operation & Summary statistics
| Node | Description |
| --- | --- |
| [Pt Argmax](pt_argmax.md) | Computes the indices of the maximum values of a PyTorch tensor along the specified dimension(s). |
| [Pt Argmin](pt_argmin.md) | Computes the indices of the minimum values of a PyTorch tensor along the specified dimension(s). |
| [Pt Max](pt_max.md) | Computes the maximum values of a PyTorch tensor along the specified dimension(s). |
| [Pt Mean](pt_mean.md) | Computes the mean of a PyTorch tensor along the specified dimension(s). |
| [Pt Median](pt_median.md) | Computes the median of a PyTorch tensor along the specified dimension(s). |
| [Pt Min](pt_min.md) | Computes the minimum values of a PyTorch tensor along the specified dimension(s). |
| [Pt Prod](pt_prod.md) | Computes the product of a PyTorch tensor along the specified dimension(s). |
| [Pt Std](pt_std.md) | Computes the standard deviation of a PyTorch tensor along the specified dimension(s). |
| [Pt Sum](pt_sum.md) | Computes the sum of a PyTorch tensor along the specified dimension(s). |
| [Pt Var](pt_var.md) | Computes the variance of a PyTorch tensor along the specified dimension(s). |
## PyTorch wrapper - Size object support
| Node | Description |
| --- | --- |
| [Pt Size](pt_size.md) | Extracts the PyTorch Size object of a PyTorch tensor using the size() method. |
| [Pt Size Create](pt_size_create.md) | Creates a PyTorch Size using values entered in the text field. |
| [Pt Size To Numpy](pt_size_to_numpy.md) | Converts PyTorch Size object to NumPy ndarray. |
| [Pt Size To String](pt_size_to_string.md) | Converts PyTorch Size object to a Python string. |
## PyTorch wrapper - Tensor creation
| Node | Description |
| --- | --- |
| [Pt Arange](pt_arange.md) | Creates a PyTorch tensor using `torch.arange` with the specified start, end, and step values. |
| [Pt Bool Create](pt_bool_create.md) | Creates a PyTorch tensor of dtype bool from True or False values entered as a list in the text field. |
| [Pt Float Create](pt_float_create.md) | Creates a PyTorch tensor with 32-bit floating point precision  |
| [Pt From Latent](pt_from_latent.md) | Casts a latent tensor as a PyTorch tensor. |
| [Pt From Numpy](pt_from_numpy.md) | Converts a NumPy ndarray to a PyTorch tensor while preserving its data type. |
| [Pt Full](pt_full.md) | Creates a PyTorch tensor filled with a specified value using the size entered in the text field. |
| [Pt Int Create](pt_int_create.md) | Creates a PyTorch tensor with 32-bit integer  |
| [Pt Linspace](pt_linspace.md) | Creates a PyTorch tensor using `torch.linspace` with the specified start, end, and steps values. |
| [Pt Ones](pt_ones.md) | Creates a PyTorch tensor of ones using the size entered in the text field. |
| [Pt Rand](pt_rand.md) | Creates a PyTorch tensor with values sampled from a uniform distribution  |
| [Pt Rand Int](pt_rand_int.md) | Creates a PyTorch tensor filled with random integers within a specified range using the size entered in the text field. |
| [Pt Randn](pt_randn.md) | Creates a PyTorch tensor with values sampled from a standard normal distribution (mean=0, std=1)  |
| [Pt Zeros](pt_zeros.md) | Creates a PyTorch tensor of zeros using the size entered in the text field. |
## PyTorch wrapper - Tensor data conversion
| Node | Description |
| --- | --- |
| [Pt To Latent](pt_to_latent.md) | Casts a PyTorch tensor as a latent tensor. |
| [Pt To Numpy](pt_to_numpy.md) | Converts PyTorch tensor to NumPy ndarray. |
| [Pt To Rgb Tensors](pt_to_rgb_tensors.md) | Splits a PyTorch tensor into R, G, and B tensors. |
## PyTorch wrapper - Training
| Node | Description |
| --- | --- |
| [Pt Data Loader](pt_data_loader.md) | Loads data from a dataset node and creates a PyTorch DataLoader.   |
| [Pt Evaluate Classification Model](pt_evaluate_classification_model.md) | Performs inference on test data and computes evaluation metrics. |
| [Pt Load Model](pt_load_model.md) | A wrapper class for saving a PyTorch model. |
| [Pt Predict Classification Model](pt_predict_classification_model.md) | Performs inference on input data. |
| [Pt Train Classification Model](pt_train_classification_model.md) | Trains a classification model using a given dataset, optimizer, and number of epochs.   |
| [Ptn Conv Model](ptn_conv_model.md) | A convolutional model consisting of multiple convolutional layers.   |
| [Ptn Linear Model](ptn_linear_model.md) | A linear model consisting of dense layers.   |
| [Ptn Resnet Model](ptn_resnet_model.md) | A Resnet model consisting of multiple Resnet layers.   |
| [Pto Adam](pto_adam.md) | Instantiates the Adam optimizer. |
| [Ptv Dataset](ptv_dataset.md) | A Torchvision Dataset class wrapper. |
| [Ptv Dataset Len](ptv_dataset_len.md) | A wrapper class that calls Python len on a dataset. |
| [Ptv Dataset Loader](ptv_dataset_loader.md) | A node to combine the dataset and data loader into a single node. |
| [Ptv Image Folder Dataset](ptv_image_folder_dataset.md) | A Torchvision ImageFolder Dataset class wrapper. |
| [Ptv Transforms Resize](ptv_transforms_resize.md) | Resizes and transforms elements of dataset to PyTorch tensors. |
| [Ptv Transforms To Tensor](ptv_transforms_to_tensor.md) | Transforms elements of dataset to PyTorch tensors. |
## PyTorch wrapper - Transform
| Node | Description |
| --- | --- |
| [Pt Flatten](pt_flatten.md) | Flattens a PyTorch tensor into a 1D tensor. |
| [Pt Permute](pt_permute.md) | Permutes the dimensions of a PyTorch tensor according to the specified order. |
| [Pt Reshape](pt_reshape.md) | Reshapes a PyTorch tensor into a specified shape using `torch.reshape()`. |
| [Pt Squeeze](pt_squeeze.md) | Removes a dimension at the specified position in the input tensor if it is of size 1. |
| [Pt Unsqueeze](pt_unsqueeze.md) | Adds a singleton dimension at the specified position in the input tensor. |
| [Pt View](pt_view.md) | Reshapes a PyTorch tensor into a specified shape using `torch.view()`. |
