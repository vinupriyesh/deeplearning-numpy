# deeplearning-numpy
Simple deeplearning library using numpy only.

## Usage

call `nn.model` with x and x with keyword parameters

### parameters
- X : input to the model - (no. of features x no. of samples)
- Y : output of the model - (no. of classes x no. of samples)

### Keyword parameters
- **alpha** : Learning Rate  
  *default* : 0.01
- **iter** : Iterations  
*default* : 3000
- **hidden_layer_dims** : Hidden layer dimentions, also decides the number of hidden layers based on the length of this list
*default* : []  
- **activation** : Activation function for the other layers and the last layer as a list of length 2.  
*supports* : sigmoid, tanh, relu, leaky_relu, softmax  
*default* : ['tanh','sigmoid']
- **batch_size** : Mini batch size  
*default* : X.shape[1]
- **dev_set_ratio** : Dev set to total data-set ratio.  
*default* : 0.02
- **parameters_file** : File-name for the parameters file to import incase of using/training a pretrained model.  
*default* : None
