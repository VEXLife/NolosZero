# -*- coding:utf-8;
# NolosZero的训练器

import consts as c
from game import game
from mctsn import *
from collections import deque
import numpy as np
import random
import os
import sys
import pickle


def calcprob(x):
    probs = x
    probs /= np.sum(probs)
    return probs


class Logger(object):
    def __init__(self, fileN=c.log_file_name):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        if "loss" in message or "learning rate" in message:
            self.log.write(message)

    def flush(self):
        self.log.flush()


class NZTrainer:
    def __init__(self):
        c.learning = True

    def loaddata(self):
        f = open(c.train_data_file_name, "rb")
        self.traindata = pickle.load(f)
        f.close()

    def savedata(self):
        f = open(c.train_data_file_name, "wb")
        pickle.dump(self.traindata, f)
        f.close()

    def extenddata(self, node, board):
        temp = np.zeros(c.edge**2)
        for a, i in node.children.items():
            temp[a] = i.N
        qpvec = c.pubNN.modifyX(board)[0]
        probsvec = calcprob(temp).reshape((c.edge, c.edge))
        self.ty.extend([[qpvec, probsvec, board.player]])
        self.ty.extend(
            [[np.transpose(qpvec), np.transpose(probsvec), board.player]])
        self.ty.extend([[np.rot90(qpvec), np.rot90(probsvec), board.player]])
        self.ty.extend([[np.flipud(qpvec), np.flipud(probsvec), board.player]])
        self.ty.extend(
            [[np.rot90(qpvec, k=2), np.rot90(probsvec, k=2), board.player]])
        self.ty.extend(
            [[np.rot90(np.flipud(qpvec)), np.rot90(np.flipud(probsvec)), board.player]])
        self.ty.extend(
            [[np.rot90(qpvec, k=3), np.rot90(probsvec, k=3), board.player]])
        self.ty.extend([[np.fliplr(qpvec), np.fliplr(probsvec), board.player]])

    def extenddata2(self, winned, winner):
        for i in self.ty:
            if winned:
                if winner == i[2]:
                    V = 1
                else:
                    V = -1
            else:
                V = 0
            self.traindata.append([i[0], i[1].reshape(c.edge**2), V])
            # print([i[0],i[1],V])
        c.gamecount += 1
        print("第{}局游戏，已产生{}组数据。".format(c.gamecount, len(self.traindata)))
        self.ty = []

    def selfplay(self):
        gm = game()
        gm.newgame()
        player = NolosZeroPlayer()
        while True:
            gm.draw()
            p = player.play(gm=gm, silence=True)
            self.extenddata(player.root_node.parent, gm)
            gm.domove(p)
            if gm.isDraw():
                # self.savedata()
                gm.draw()
                self.extenddata2(False, None)
                if len(self.traindata) >= c.minbatch:
                    return self.traindata
                gm.newgame()
                continue
            winned, winner = gm.isWin(p)
            if winned:
                # self.savedata()
                gm.draw()
                self.extenddata2(winned, winner)
                if len(self.traindata) >= c.minbatch:
                    return self.traindata
                gm.newgame()
                continue

    def train(self):
        if not os.path.exists(c.train_data_file_name):
            self.traindata = []
            self.savedata()
        else:
            self.loaddata()
        self.traindata = deque(maxlen=c.buffer_size)
        self.ty = []
        try:
            while True:
                inputdata = random.sample(self.selfplay(), c.minbatch)
                print("数据量已达最小批训练数据量大小，开始训练。")
                x, probs, values = [], [], []
                for data in inputdata:
                    x.append(data[0])
                    probs.append(data[1])
                    values.append(data[2])
                c.pubNN.model.fit(np.array(x), [np.array(probs), np.array(
                    values)], batch_size=c.minbatch, epochs=c.epochs)
                c.pubNN.save_net()
                print("本次训练完成！")
        except KeyboardInterrupt:
            self.savedata()


sys.stdout = Logger()
NZTrainer().train()
