# Neural Networks

A collection of neural network implementations using pure python for
education purposes. This is not in a million years intended to be a 
replacement for Tensorflow or Pytorch.

# `nn.py`

This script contains a very simple implementation of a feed-forward
neural network including the sigmoid function, squared error loss
function, their derivatives, and a quick main function to demonstrate
the network in action.

The network is implemented using more sophisticated dynamic programming
and matrix operations to achieve greater performance over the more naive
implementations which computes the weight gradients on a per-weight
basis.

Overall, by the few experiments performed, the back propagation
algorithms seems to be working as intended. This was by far the most
challenging implementation detail as the mathematics behind the
algorithm is quite complicated in terms of notation. However, the
rest of the implementation was rather simple if not trivial.

By implementing such a network completely from scratch, without the use
of third party libraries such as `tensorflow` or `pytorch`, deeper
insight can be gained into the internal workings of the network that
are not frequently discussed in books, tutorials, or online blogs. In
fact, the reason why back propagation was as challenging to implement
as it was was simply due to the lack of deeper explanation behind the
mathematics of it. There is a great amount of articles describing the
abstract mathematics behind it, but never how it is actually implemented.

Moreover, by studying the structure of a neural network built from scratch,
questions such as "why are the gradients so tiny?" or "why is the network
taking 50000 epochs to converge?" are quite simple to answer because
one understands exactly how the operations work.

(I will add more detail to the derivation and concrete implementation of
the network to alleviate my own grievances with the state of ML
education on the internet sometime in the future.)