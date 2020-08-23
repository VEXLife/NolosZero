# -*- coding:utf-8;
# NolosZero的AlphaZero玩家

import consts as c
import numpy as np
from copy import deepcopy


def softmax(x):
    probs = np.exp(x)
    probs /= np.sum(probs)
    return probs


class Leaf:
    def __init__(self, parent=None, predP=1):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.Q = 0
        self.P = predP

    def isroot(self):
        return self.parent == None

    def isleaf(self):
        return len(self.children) == 0

    def update(self, val):
        if not self.isroot():
            self.parent.update(-val)
        self.N += 1
        self.Q += 1.0*(val-self.Q)/self.N

    def W(self):
        self.U = (c.c_puct+np.log(1+(1+self.parent.N)/c.c_base)) * \
            self.P*np.sqrt(self.parent.N)/(self.N+1)
        return self.Q+self.U

    def select(self):
        return max(self.children.items(), key=lambda act_node: act_node[1].W())

    def expand(self, actions):
        for action, prob in actions:
            if action not in self.children:
                self.children[action] = Leaf(self, prob)


class NolosZeroPlayer:
    root_node = Leaf()

    def MCTSemu(self, lim):
        for i in range(lim):
            board = deepcopy(self.state)
            node = self.root_node
            action = -1
            while not node.isleaf():
                action, node = node.select()
                board.domove(action)
            actprobs, V = c.pubNN.predict(board)
            if action == -1:
                winned = False
            else:
                winned, winner = board.isWin(action)
            if winned:
                V = 1 if winner == board.player else -1
            elif board.isDraw():
                V = 0
            else:
                node.expand([(i, actprobs[i]) for i in board.availablemoves])
            node.update(-V)

    def play(self, gm, silence=False):
        self.state = gm
        self.root_node=Leaf()
        if not silence:
            print("NolosZero正在思考中...")
        self.MCTSemu(c.NZtimes)
        if c.learning:
            probs = softmax(
                1.0/c.tau*np.log([i.N for i in self.root_node.children.values()]))
            action = np.random.choice([a for a in self.root_node.children.keys(
            )], p=c.epsilon*probs+(1-c.epsilon)*np.random.dirichlet([1 for j in probs]))
        else:
            probs = softmax(np.log([i.N for i in self.root_node.children.values()]))
            print(probs)
            action = list(self.root_node.children.keys())[np.argmax(probs)]
        self.root_node = self.root_node.children[action]
        return action
