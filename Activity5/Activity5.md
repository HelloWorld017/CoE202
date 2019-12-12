# Activity 5
CoE202(A) / 20190146 Kim Yohan

## Problem A
> Use GRU, LSTM and Simple RNN functions for training . Compare each of results.

* GRU  
`cells = tf.contrib.rnn.GRUCell(num_units = 128)`
* LSTM  
`cells = tf.contrib.rnn.BasicLSTMCell(num_units = 128)`
* Basic RNN  
`cells = tf.contrib.rnn.BasicRNNCell(num_units = 128)`

|           |   GRU   |   LSTM   | Simple RNN |
|:---------:|:-------:|:--------:|:----------:|
| Train Acc | 0.658221| 0.620404 |  0.617937  |
| Test Acc  | 0.635929| 0.611418 |  0.591541  |

The GRU was the best performing and the next was LSTM and Simple RNN.
Actually, the accuracy of LSTM was keep converged to a certain value,
so I have changed learning rate to 0.001 and trained multiple time.

## Problem B
> Replace the RNN with DNN as below.

```py
x = tf.reshape(X, [-1, max_sequence_length * n_input])

w_init = tf.variance_scaling_initializer()
b_init = tf.constant_initializer(0.)

## 1st hidden layer
w1 = tf.get_variable('weight1', [max_sequence_length * n_input, 256], initializer = w_init)  # weight for 1st hidden layer which have 256 units
b1 = tf.get_variable('biases1', [256], initializer = b_init)                                 # bias for 1st hidden layer which have 256 units
h  = tf.matmul(x, w1) + b1                                                                   # matrix multiplication
h  = tf.nn.relu(h)                                                                           # relu activation

## 2nd hidden layer
w2 = tf.get_variable('weight2', [256, 256], initializer = w_init)                            # weight for 2nd hidden layer which have 256 units
b2 = tf.get_variable('biases2', [256], initializer = b_init)                                 # bias for 2nd hidden layer which have 256 units
h  = tf.matmul(h, w2) + b2                                                                   # matrix multiplication
h  = tf.nn.relu(h)                                                                           # relu activation

## output layer
w3 = tf.get_variable('weight3', [256, 256], initializer = w_init)                            # weight for output layer which have 256 units

y_ = tf.matmul(h, w3)
```
I have implemented the modified networks by adding 2 hidden layers with 256 units
and using ReLU as an activation function.

|           |   DNN    |
|:---------:|:--------:|
| Train Acc | 0.979073 |
| Test Acc  | 0.780003 |

And the modified network had generally better performance.
I think it is because the sequences weren't too long and
could be learned well by only using DNN.
