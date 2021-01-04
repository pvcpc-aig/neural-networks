import os
import sys
import math

import numpy as np


def sigmoid(x):
    # ok so neat thing I found out while experimenting was that
    # modifying the hard coded "bias" of an activation function whose
    # derivative quickly becomes zero makes the network much slower
    # to learn with a static learning rate because it has to inch
    # towards the minimum on a magnitude of 10^-10 each step. However,
    # with the right bias, the network is much much much more accurate.
    # Fascinating.
    return 1 / (1 + math.exp(-x + 12))

def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)


def vector_sigmoid(X):
    out = np.zeros([len(X)])
    for i, x in enumerate(X):
        out[i] = sigmoid(x)
    return out

def vector_sigmoid_prime(X):
    out = np.zeros([len(X), len(X)])
    for i, x in enumerate(X):
        out[i][i] = sigmoid_prime(x)
    return out


def squared_error_loss(X, Y):
    min_len = min(len(X), len(Y))
    out = 0
    for i in range(min_len):
        diff = X[i] - Y[i]
        out += diff * diff
    return out

def squared_error_loss_prime(X, Y):
    min_len = min(len(X), len(Y))
    out = np.zeros([min_len])
    for i in range(min_len):
        out[i] = 2 * (X[i] - Y[i])
    return out


class NeuralNetwork:

    def __init__(self, layer_desc, loss_func, loss_deriv):
        """
        :param layer_desc
            A list of tuples describing individual layers in the given
            format:

            (
                <activation function>, 
                <derivative function>,
                <size>
            )

            This includes the input layer, and it must be the first one
            in the form:

            (None, None, <size>)

            The activation and derivative of the activation functions
            must accept parameters in the order:

            (<Z vector>)

            The activation function must return a vector of the same size
            as the Z vector, and the derivative function must return a 
            Jacobian matrix of partial derivatives of the activation
            function with respect to the Z vector in the form:

                Row 1: [ dA1/dZ1 ... dA1/dZr ]
                Row 2: [ dA2/dZ1 ... dA2/dZr ]
                ...
                Row c: [ dAc/dZ1 ... dAc/dZr ]
            
            i.e., Jacobian standard form.

        :param loss_func
            A stateless loss function that accepts:
            
            (<prediction vector>, <actual vector (Y)>)

            and returns a real number indicating the loss of the
            network.
        
        :param loss_deriv
            The stateless derivative of `loss_func` which accepts:

            (<prediction vector>, <actual vector (Y)>)

            and returns the gradient vector.
        """
        self.loss_func = loss_func
        self.loss_deriv = loss_deriv

        # the "width" is the number of layers in the network, including
        # the input layer.
        self.width = len(layer_desc)

        # the total number of neurons (excluding input neurons) in
        # the network.
        self.nneurons = 0
        
        # Tracks epochs during training.
        self.epoch = 0

        # Tracks number of samples trained on per epoch.
        self.sample = 0

        # The summation of the loss function over the course of an epoch.
        self.errsum = 0

        # Ds - Dimension vector of each network layer.
        self.Ds = np.zeros((self.width))
        # AFs - Activation functions corresponding to their layer.
        self.AFs = [ None ] * self.width
        # DFs - Derivative functions corresponding to their layer.
        self.DFs = [ None ] * self.width

        # Array of weight matrices per layer.
        self.Ws = []
        self.dWs = []
        # Matrix of row vectors which are the inputs to the activation
        # function corresponding to its layer. These are computed as
        # the weight matrix mulitplied with the input vector.
        self.Zs = []
        # Outputs of the neuron layers, (As for Activations).
        self.As = []
        # Outputs of the derivatives with respect to Z vector.
        self.dAdZs = []

        for i, x in enumerate(layer_desc):
            act_fun, der_fun, size = x
            self.Ds[i] = size
            self.AFs[i] = act_fun
            self.DFs[i] = der_fun

            if i > 0:
                prior = layer_desc[i - 1]
                _, _, prior_size = prior
                self.nneurons += size
                self.Ws.append(np.zeros([size, prior_size]))
                self.dWs.append(np.zeros([size, prior_size]))
                self.Zs.append(np.zeros([size]))
                self.dAdZs.append(np.zeros([size, size]))
            else:
                self.Ws.append(None)
                self.dWs.append(None)
                self.Zs.append(None)
                self.dAdZs.append(None)

            self.As.append(np.zeros([size]))

    # network config
    def network_reset_error(self):
        self.errsum = 0
    
    # weight config 
    def weights_const(self, w_0):
        for i in range(1, self.width):
            self.Ws[i].fill(w_0)
    
    def weights_inv_n(self):
        self.weights_const(1 / self.nneurons)
    
    # training
    def train_feed(self, X, Y, train_rate=1e-2):
        lX = len(X)
        lY = len(Y)
        if self.Ds[0] != lX:
            raise ValueError("Invalid input vector dimensions")
        if self.Ds[-1] != lY:
            raise ValueError("Invalid output vector dimensions")

        # configure input neurons
        for i, x in enumerate(X):
            self.As[0][i] = x

        # feed forward, i = layer
        for i in range(1, self.width):
            Di_1 = self.Ds[i - 1] #<vector
            Ai_1 = self.As[i - 1] #<vector

            AFi = self.AFs[i] #< function
            DFi = self.DFs[i] #< function

            Di = self.Ds[i] #< vector
            Wi = self.Ws[i] #< matrix
            Zi = self.Zs[i] #< vector
            Ai = self.As[i] #< vector
            dAdZi = self.dAdZs[i] #< matrix

            np.matmul(Wi, Ai_1, out=Zi)
            np.copyto(Ai, AFi(Zi))
            np.copyto(dAdZi, DFi(Zi))
        
        # the last layer output
        An = self.As[-1]
            
        # compute error
        loss = self.loss_func(An, Y)
        loss_grad = self.loss_deriv(An, Y)
        self.errsum += loss

        print("[Epoch %d] Loss on sample %d: %.3f" % 
            (self.epoch, self.sample, loss))
        
        self.sample += 1
        
        # back propagate on the loss. 
        i = self.width - 1
        delta = loss_grad # stores backpropagation evaluation; always a vector
        while i > 0:
            Ai_1 = self.As[i - 1]

            Di = self.Ds[i]
            Wi = self.Ws[i]
            dWi = self.dWs[i]
            dAdZi = self.dAdZs[i]

            np.matmul(np.transpose(dAdZi), delta, out=delta)

            # get gradient matrix with respect to layer i
            for j in range(len(delta)): # 'optimized' tensor * vector
                dWi[j] = Ai_1 * delta[j] # this is also a de-facto transpose
            
            # apply chain rule
            delta = np.matmul(np.transpose(Wi), delta)

            # go back a layer
            i -= 1
        
        # adjust the weights based on the gradients
        for i in range(1, self.width):
            self.Ws[i] -= self.dWs[i] * train_rate
        
        # return the prediction vector
        return self.As[-1]
    
    def train_complete_epoch(self):
        print("Epoch %d complete with %d samples, cumulative error: %.3f" 
            % (self.epoch, self.sample, self.errsum))
        self.sample = 0
        self.epoch += 1
        self.errsum = 0


if __name__ == "__main__":
    nn = NeuralNetwork([
        (None, None, 1),
        (vector_sigmoid, vector_sigmoid_prime, 1)
    ], 
    squared_error_loss, squared_error_loss_prime)

    nn.weights_inv_n()

    training = [
        ([1], [0]),
        ([2], [0]),
        ([3], [0.5]),
        ([4], [1]),
        ([5], [1]),
    ]

    for epoch in range(20000):
        for X, Y in training:
            nn.train_feed(X, Y)
            # print("Gradient:")
            # print(nn.dWs[1])
        nn.train_complete_epoch()
    
    print("Final weights:")
    print(nn.Ws[1])
