# -*- coding: utf-8 -*-
"""
We construct a deep neural network, which we train using the deep Galerkin method (DGM) with L-infinity loss.

The DGM is used to solve high-dimensional PDEs, in the form of the Hamilton-Jacobi-Bellman (HJB) equation
associated with the Mean Field Control Problem (MFCP).

"""

import tensorflow as tf

"""# Model"""

# LSTM-like layer used in DGM - modification of keras Layer class
class LSTMLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, units, trans="tanh"):
        '''
        Args:
            units (int): number of units in each layer
            trans (str): nonlinear activation function
                         one of: "tanh" (default), "relu", or "sigmoid"
        Returns:
            customized keras Layer object used as intermediate layers in DGM

        '''

        # create an instance of a keras Layer object (call initialize function of superclass of LSTMLayer)
        super().__init__()
        self.units = units

        if trans == "tanh":
            self.trans = tf.nn.tanh
        elif trans == "relu":
            self.trans = tf.nn.relu
        elif trans == "sigmoid":
            self.trans = tf.nn.sigmoid

    # define LSTMLayer parameters
    def build(self, input_shape):

        # U matrix (weighting for original inputs X)
        self.Uz = self.add_weight(name="Uz", shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Ug = self.add_weight(name="Ug", shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Ur = self.add_weight(name="Ur", shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Uh = self.add_weight(name="Uh", shape=(input_shape[-1], self.units),
                                  initializer='random_normal',
                                  trainable=True)

        # super().build(input_shape)

        # W matrix (weighting for outputs from previous layer)
        self.Wz = self.add_weight(name="Wz", shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Wg = self.add_weight(name="Wg", shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Wr = self.add_weight(name="Wr", shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True)
        self.Wh = self.add_weight(name="Wh", shape=(self.units, self.units),
                                  initializer='random_normal',
                                  trainable=True)

        # bias vector
        self.bz = self.add_weight(name="bz", shape=(self.units,),
                                  initializer='random_normal',
                                  trainable=True)
        self.bg = self.add_weight(name="bg", shape=(self.units,),
                                  initializer='random_normal',
                                  trainable=True)
        self.br = self.add_weight(name="br", shape=(self.units,),
                                  initializer='random_normal',
                                  trainable=True)
        self.bh = self.add_weight(name="bh", shape=(self.units,),
                                  initializer='random_normal',
                                  trainable=True)

    # main function to be called
    def call(self, X, S):
        '''Compute output of a LSTMLayer for given inputs X and S.
        Args:
            X: data input
            S: output of previous layer

        Returns:
            S_new: input to next LSTMLayer
        '''

        # compute components of LSTMLayer output
        Z = self.trans(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.trans(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.trans(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))

        # compute LSTMLayer outputs
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))
        return S_new


# Fully connected (dense) layer - modification of keras Layer class
class DenseLayer(tf.keras.layers.Layer):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, units, trans=None):
        '''
        Args:
            units (int): number of units in each layer
            trans (str): nonlinear activation function
                         one of: "tanh", "relu", "sigmoid", or None (default)
                         None means identity map
        Returns:
            customized keras (fully connected) Layer object
        '''

        # create an instance of a keras Layer object (call initialize function of superclass of DenseLayer)
        super().__init__()
        self.units = units

        if trans:
            if trans == "tanh":
                self.trans = tf.tanh
            elif trans == "relu":
                self.trans = tf.nn.relu
            elif trans == "sigmoid":
                self.trans = tf.nn.sigmoid
        else:
            self.trans = trans

    # define DenseLayer parameters
    def build(self, input_shape):

        # W matrix (weighting for outputs from previous layer)
        self.W = self.add_weight(name="W", shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

        # bias vector
        self.b = self.add_weight(name="b", shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    # main function to be called
    def call(self, X):
        '''Compute output of a DenseLayer for a given input X.
        Args:
            X: input to layer
        Returns:
            S: input to next layer
        '''

        # compute DenseLayer output
        S = tf.add(tf.matmul(X, self.W), self.b)

        if self.trans:
            S = self.trans(S)
        return S


# Neural network architecture used in DGM - modification of keras Model class
class DGMNet(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, units, n_layers, final_trans=None):
        '''
        Args:
            units (int):       number of units in each layer
            n_layers (int):    number of intermediate LSTM layers
            final_trans (str): nonlinear activation function used in final layer
                               one of: "tanh" (default), "relu", or "sigmoid"
        Returns:
            customized keras Model object representing DGM neural network
        '''

        # create an instance of a keras Model object (call initialize function of superclass of DGMNet)
        super().__init__()

        # define initial layer as fully connected
        self.initial_layer = DenseLayer(units, trans="tanh")

        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(units, trans="tanh"))

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, trans=final_trans)

    # main function to be called
    def call(self, x, t):
        '''Run the DGM model and obtain fitted function value at the inputs (t,x).
        Args:
            x: sampled space inputs
            t: sampled time inputs
        Returns:
            result: fitted function value
        '''

        # define initial inputs as time-space pairs
        X = tf.concat([x, t], 1)

        # call initial layer
        initial = self.initial_layer
        S = initial(X)

        # call intermediate LSTM layers
        for i in range(self.n_layers):
            LSTM = self.LSTMLayerList[i]
            S = LSTM(X, S)

        # call final layer
        final = self.final_layer
        result = final(S)
        return result