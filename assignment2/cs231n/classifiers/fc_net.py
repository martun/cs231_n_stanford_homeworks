from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        activations1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(activations1, self.params['W2'], self.params['b2'])
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        
        # Add regularization loss.
        loss += self.reg / 2 * (np.sum(self.params['W1'] * self.params['W1']) + 
                            np.sum(self.params['W2'] * self.params['W2']))
        
        dactivation2, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        dinput, grads['W1'], grads['b1'] = affine_relu_backward(dactivation2, cache1)
        
        # Add regularization gradient.
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        for i, dim in enumerate(hidden_dims):
            if self.normalization=='batchnorm':
                self.params['gamma%d' % (i+1)] = np.ones(dim)
                self.params['beta%d' % (i+1)] = np.zeros(dim)
            # Cannot initialize layer normalization Gamma and Beta, because don't know the batch sizes to be used.
            #elif self.normalization=='layernorm':
            #    self.params['gamma%d' % (i+1)] = np.ones(dim)
            #    self.params['beta%d' % (i+1)] = np.zeros(dim)
            if i == 0:
                self.params['b1'] = np.zeros(dim)
                self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, dim))
            else:
                self.params['b%d' % (i+1)] = np.zeros(dim)
                self.params['W%d' % (i+1)] = np.random.normal(0.0, weight_scale, (hidden_dims[i - 1], dim))
        
        self.params['b%d' %  (self.num_layers)] = np.zeros(num_classes)
        self.params['W%d' %  (self.num_layers)] = np.random.normal(0.0, weight_scale, (hidden_dims[-1], num_classes))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        
        # Doing these reshapes for Dropout notebook.
        N = X.shape[0]
        D = int(np.prod(X.shape) / N)
        X = np.reshape(X, (N, D))
        
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        elif self.normalization=='layernorm':
            # Initialize gamma and beta sizes.
            if not self.params['gamma%d' % (i+1)]:
                self.params['gamma%d' % (i+1)] = np.ones(N)
            if not self.params['beta%d' % (i+1)]:
                self.params['beta%d' % (i+1)]= np.zeros(N)
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        
        activations = [None] * self.num_layers
        affine_cache = [None] * self.num_layers
        dropout_cache = [None] * self.num_layers
        relu_cache = [None] * self.num_layers
        batchnorm_cache = [None] * self.num_layers
        # Run through first L-1 layers.
        for i in range(self.num_layers - 1):
            if i == 0:
                activations[0], affine_cache[0] = affine_forward(X, self.params['W1'], self.params['b1'])
            else:
                activations[i], affine_cache[i] = affine_forward(
                    activations[i - 1], self.params['W%d' % (i+1)], self.params['b%d' % (i+1)])
            if self.normalization=='batchnorm':
                activations[i], batchnorm_cache[i] = batchnorm_forward(
                    activations[i], self.params['gamma%d' % (i+1)], self.params['beta%d' % (i+1)], self.bn_params[i])
            elif self.normalization=='layernorm':
                activations[i], batchnorm_cache[i] = layernorm_forward(
                    activations[i], self.params['gamma%d' % (i+1)], self.params['beta%d' % (i+1)], self.bn_params[i])
            activations[i], relu_cache[i] = relu_forward(activations[i])
            if self.use_dropout:
                activations[i], dropout_cache[i] = dropout_forward(activations[i], self.dropout_param)
        
        # Run the last affine layer and SoftMax.
        activations[self.num_layers - 1], affine_cache[self.num_layers - 1] = affine_forward(
            activations[self.num_layers - 2],
            self.params['W%d' % (self.num_layers)],
            self.params['b%d' % (self.num_layers)])
        
        scores = activations[self.num_layers - 1]
               
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        
        # Add regularization loss.
        for i in range(self.num_layers):
            loss += self.reg / 2 * (np.sum(self.params['W%d' % (i+1)] * self.params['W%d' % (i+1)]))
        
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        
        # Compute the gradients.
        dinputs, grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = affine_backward(
            dscores, affine_cache[self.num_layers - 1])
        grads['W%d' % self.num_layers] += self.reg * self.params['W%d' % self.num_layers]
        
        # Backprop through first L-1 layers.
        for i in reversed(range(self.num_layers - 1)):
            if self.use_dropout:
                dinputs = dropout_backward(dinputs, dropout_cache[i])
            dinputs = relu_backward(dinputs, relu_cache[i])
            
            if self.normalization=='batchnorm':
                dinputs, grads['gamma%d' % (i+1)], grads['beta%d' % (i+1)] = batchnorm_backward(dinputs, batchnorm_cache[i])
            elif self.normalization=='layernorm':
                dinputs, grads['gamma%d' % (i+1)], grads['beta%d' % (i+1)] = layernorm_backward(dinputs, batchnorm_cache[i])
                
            dinputs, grads['W%d' % (i + 1)], grads['b%d' % (i + 1)] = affine_backward(dinputs, affine_cache[i])
            grads['W%d' % (i + 1)] += self.reg * self.params['W%d' % (i + 1)]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
