# Quassi_RNN
Tensorflow implementation of Quassi RNN
 This is an implementation of the QuasiRNN proposed in the [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576). The cell is implemented from scratch as a part of learning experience. QuasiRNN are yet another architecture for RNN (and its popular extensions like LSTM, GRU etc), but at the same time are less time consuming than LSTMs etc without compromising the performance. It uses a convolutional layer, which can be applied to various timesteps in parallel and a minimal recurrent pooling layer that applies in parallel acros timesteps. The authors have shown this works for Sequence modelling tasks like Neural Translation. This implementatyion explores its applications on Vision. Currently, this is written for digit recognition (8X8 version in scikit dataset). 
# Requirements
* [Python >2.7](https://www.python.org/downloads/)
* [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
* [Numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [Scikit-learn](http://scikit-learn.org/stable/install.html)

# Running the system

<code> python quasi_rnn.py </code>

# To Do list
* Tune the model.
* Run it on MNIST, CIFAR datasets
* Convert it into a plug in module


Feel free to submit your comments and interesting PR's.
