# -*- coding:utf-8;
#NolosZero的训练器

import consts as c
from game import game
from mctsn import *
from keras.callbacks import *
from collections import deque
import keras.backend as K
import numpy as np
import random
import os
import sys
import pickle

def softmax(x):
    probs=np.exp(x-np.max(x))
    probs/=np.sum(probs)
    return probs

class Logger(object):
    def __init__(self,fileN = c.log_file_name):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
        
    def write(self,message):
        self.terminal.write(message)
        if "loss" in message or "learning rate" in message:
            self.log.write(message + "当前学习率：" + str(float(K.get_value(c.pubNN.model.optimizer.lr))) +"\n")

    def flush(self):
        self.log.flush()

class NZTrainer:
    def __init__(self):
        c.learning=True
        #self.rlr=ReduceLROnPlateau(monitor='loss',factor=c.factor,patience=c.patience,verbose=1,mode="auto",min_delta=c.min_delta,cooldown=0,min_lr=c.min_lr)

    def loaddata(self):
        f=open(c.train_data_file_name,"rb")
        self.traindata=pickle.load(f)
        f.close()

    def savedata(self):
        f=open(c.train_data_file_name,"wb")
        pickle.dump(self.traindata,f)
        f.close()

    def rotdata(self,data):
        edata=[]
        for i in [1, 2, 3, 4]:
            ex = np.array([np.rot90(s, i) for s in data[0][0]])
            ey = np.rot90(data[1].reshape(c.edge, c.edge), i)
            edata.append((ex,ey.flatten(),data[2]))
            ex = np.array([np.fliplr(s) for s in data[0][0]])
            ey = np.fliplr(ey)
            edata.append((ex,ey.flatten(),data[2]))
        return edata

    '''def rotdata(self,data):
        edata=[]
        for i in [1, 2, 3, 4]:
            ex = np.array([np.rot90(s, i) for s in data[0][0]])
            ey = np.rot90(np.flipud(data[1].reshape(c.edge, c.edge)), i)
            edata.append((ex,np.flipud(ey).flatten(),data[2]))
            ex = np.array([np.fliplr(s) for s in data[0][0]])
            ey = np.fliplr(ey)
            edata.append((ex,np.flipud(ey).flatten(),data[2]))
        return edata'''

    def extenddata(self,node,board):
        temp=np.zeros(c.edge**2)
        for a,i in node.children.items():
            temp[a]=i.N
        self.ty.extend(self.rotdata([c.pubNN.modifyX(board),softmax(1.0/c.tau*np.log(temp)),board.player]))

    def extenddata2(self,winned,winner):
        for i in self.ty:
            V=0
            if winned:
                if winner==i[2]:
                    V=1
                else:
                    V=-1
            else:
                V=0
            self.traindata.append([i[0],i[1],V])
            #print([i[0],i[1],V])
        c.gamecount+=1
        print("第{}局游戏，已产生{}组数据。".format(c.gamecount,len(self.traindata)))
        self.ty=[]

    def selfplay(self):
        gm=game()
        gm.newgame()
        c.tau=1
        player=NolosZeroPlayer()
        while True:
            #gm.draw()
            p=player.play(gm=gm,silence=True)
            self.extenddata(player.rn.parent,deepcopy(gm))
            gm.domove(p)
            #if(c.edge**2-len(gm.availablemoves)==c.tempnew): c.tau=1e-3
            if gm.isDraw():
                #self.savedata()
                #gm.draw()
                self.extenddata2(False,None)
                if len(self.traindata)>=c.minbatch:
                    return self.traindata
                gm.newgame()
                player=NolosZeroPlayer()
                continue
            winned,winner=gm.isWin(p)
            if winned:
                #self.savedata()
                #gm.draw()
                self.extenddata2(winned,winner)
                if len(self.traindata)>=c.minbatch:
                    return self.traindata
                gm.newgame()
                player=NolosZeroPlayer()
                continue

    def train(self):
        if not os.path.exists(c.train_data_file_name):
            self.traindata=[]
            self.savedata()
        else:
            self.loaddata()
        self.traindata=deque(maxlen=c.buffer_size)
        self.ty=[]
        try:
            while True:
                inputdata=random.sample(self.selfplay(),c.minbatch)
                print("数据量已达最小批训练数据量大小，开始训练。")
                x,probs,values=[],[],[]
                for data in inputdata:
                    x.append(data[0])
                    probs.append(data[1])
                    values.append([2])
                old_probs, old_v = c.pubNN.model.predict_on_batch(np.array(x))
                for i in range(c.epochs):
                    K.set_value(c.pubNN.model.optimizer.lr,c.lr*c.factor)
                    c.pubNN.model.fit(np.array(x),[np.array(probs),np.array(values)],batch_size=c.minbatch,epochs=1)
                    new_probs, new_v = c.pubNN.model.predict_on_batch(np.array(x))
                    kl = np.mean(np.sum(old_probs * (
                            np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                    )
                if kl > c.min_delta * 2 and c.factor > 0.1:
                    c.factor /= 1.5
                elif kl < c.min_delta / 2 and c.factor < 10:
                    c.factor *= 1.5
                c.pubNN.save_net()
                print("本次训练完成！")
        except KeyboardInterrupt:
            self.savedata()
sys.stdout=Logger()
NZTrainer().train()