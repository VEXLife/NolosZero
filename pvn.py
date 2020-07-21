# -*- coding:utf-8;
#NolosZero神经网络

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
#import pickle
import consts as c
import os
import numpy as np
from copy import deepcopy

class CNN:
    def __init__(self):
        if os.path.exists(c.model_file_name):
            self.load_net()
        else:
            self.construct()
        print(self.model.summary())
    
    def construct(self):
        inputx=base=Input((c.planecount,c.edge,c.edge))
        base=Conv2D(filters=c.filters[0],kernel_size=[3,3],padding="same",data_format="channels_first",activation="relu",kernel_regularizer=l2(c.l2c))(base)#卷积部分预处理
        base=Conv2D(filters=c.filters[1],kernel_size=[3,3],padding="same",data_format="channels_first",activation="relu",kernel_regularizer=l2(c.l2c))(base)#残差块
        base=Conv2D(filters=c.filters[2],kernel_size=[3,3],padding="same",data_format="channels_first",activation="relu",kernel_regularizer=l2(c.l2c))(base)#残差块
        pnet=Conv2D(filters=c.filters[3],kernel_size=[1,1],data_format="channels_first",activation="relu",kernel_regularizer=l2(c.l2c))(base)#策略网络
        pnet=Flatten()(pnet)
        self.pnet=Dense(c.edge**2,activation="softmax",kernel_regularizer=l2(c.l2c))(pnet)
        vnet=Conv2D(filters=c.filters[4],kernel_size=[1,1],data_format="channels_first",activation="relu",kernel_regularizer=l2(c.l2c))(base)#价值网络
        vnet=Flatten()(vnet)
        vnet=Dense(c.FCs[0],kernel_regularizer=l2(c.l2c))(vnet)
        self.vnet=Dense(c.FCs[1],activation="tanh",kernel_regularizer=l2(c.l2c))(vnet)
        self.model=Model(inputx,[self.pnet,self.vnet])
        self.model.compile(optimizer=Adam(lr=c.lr),loss=["categorical_crossentropy","mean_squared_error"])
        #self.load_net()

    def load_net(self):
        self.model=load_model(c.model_file_name)

    def save_net(self):
        self.model.save(c.model_file_name)

    def modifyX(self,gm):
        if gm.player==c.player1:
            plane1=gm.plane1
            plane2=gm.plane2
            plane3=deepcopy(gm.plane_e)
            plane4=deepcopy(gm.plane_e)
        else:
            plane1=gm.plane2
            plane2=gm.plane1
            plane3=deepcopy(gm.plane_f)
            plane4=deepcopy(gm.plane_e)
        if not gm.lastmove==-1:
            plane4[gm.lastmove]=1
        x=np.array((plane1,plane2,plane3,plane4)).reshape(1,c.planecount,c.edge,c.edge)
        #print(x)
        return x

    def predict(self,gm):
        x=self.modifyX(gm)
        val1,val2=self.model.predict(x)
        #print(val1)
        return val1,val2[0][0]