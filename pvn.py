# -*- coding:utf-8;
# NolosZero的神经网络

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
from keras.activations import *
import consts as c
import os
import numpy as np
from copy import deepcopy
from keras.utils import plot_model


class CNN:
    def __init__(self):
        if os.path.exists(c.model_file_name):
            self.load_net()
        else:
            self.construct()
        print(self.model.summary())

    def construct(self):
        inputx = Input(shape=(c.edge, c.edge))
        base = Reshape((c.edge, c.edge) + (1,))(inputx)
        for head_filter in c.filters["head"]:
            base = Conv2D(head_filter, 3, padding="same", kernel_regularizer=l2(
                c.l2c), bias_regularizer=l2(c.l2c))(base)
            base = BatchNormalization()(base)
            base = ReLU()(base)
        for residual_filters in c.filters["residual"]:
            side = base
            for i, residual_filter in enumerate(residual_filters):
                side = Conv2D(residual_filter, 3, padding="same", kernel_regularizer=l2(
                    c.l2c), bias_regularizer=l2(c.l2c))(side)
                side = BatchNormalization()(side)
                if(i == len(residual_filters)-1):
                    base = Add()([base, side])
                    base = ReLU()(base)
                    break
                side = ReLU()(side)
        for policy_filter in c.filters["policy"]:
            pnet = Conv2D(policy_filter, 3, padding="same", kernel_regularizer=l2(
                c.l2c), bias_regularizer=l2(c.l2c))(base)
            pnet = BatchNormalization()(pnet)
            pnet = ReLU()(pnet)
        pnet = Conv2D(1, 3, padding="same", kernel_regularizer=l2(
            c.l2c), bias_regularizer=l2(c.l2c))(pnet)
        pnet = Flatten()(pnet)
        pnet = Softmax()(pnet)
        vnet = Conv2D(1, 3, padding="same", kernel_regularizer=l2(
            c.l2c), bias_regularizer=l2(c.l2c))(base)
        vnet = BatchNormalization()(vnet)
        vnet = ReLU()(vnet)
        vnet = Flatten()(vnet)
        '''
        vnet=Dense(8,activation=tanh,kernel_regularizer=l2(c.l2c),bias_regularizer=l2(c.l2c))(vnet)
        vnet=ReLU()(vnet)
        '''
        vnet = Dense(1, activation=tanh, kernel_regularizer=l2(
            c.l2c), bias_regularizer=l2(c.l2c))(vnet)
        self.model = Model(inputx, [pnet, vnet])
        plot_model(self.model, to_file=c.model_plot_file_name,
                   show_shapes=True)
        self.model.compile(optimizer=Adam(lr=c.lr), loss=[
                           "categorical_crossentropy", "mean_squared_error"])

    def load_net(self):
        self.model = load_model(c.model_file_name)

    def save_net(self):
        self.model.save(c.model_file_name)

    def modifyX(self, gm):
        if gm.player == c.player1:
            plane = np.array(gm.plane)
        else:
            plane = -1*np.array(gm.plane)
        x = np.array(plane).reshape(1, c.edge, c.edge)
        return x

    def predict(self, gm):
        x = self.modifyX(gm)
        val1, val2 = self.model.predict(x)
        return val1[0], val2[0][0]
