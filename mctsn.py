# -*- coding:utf-8;
#NolosZero的神经网络指导的MCTS模拟

import consts as c
import numpy as np
from copy import deepcopy

def softmax(x):
    probs=np.exp(x-np.max(x))
    probs/=np.sum(probs)
    return probs

class Leaf:
    def __init__(self,parent=None,predP=1):
        self.parent=parent
        self.children={}        
        self.N=0
        self.Q=0
        self.P=predP

    def isroot(self):
        return self.parent==None

    def isleaf(self):
        return len(self.children)==0

    def update(self,val):
        if not self.isroot():
            self.parent.update(-val)
        self.N+=1
        self.Q+=1.0*(val-self.Q)/self.N
            
    def W(self):
        self.U=c.c_puct*self.P*np.sqrt(self.parent.N)/(self.N+1)
        return self.Q+self.U

    def select(self):
        return max(self.children.items(),key=lambda act_node: act_node[1].W())

    def expand(self,actions):
        for action,prob in actions:
            if action not in self.children:
                self.children[action] = Leaf(self,prob)

class NolosZeroPlayer:
    rn=None
    def MCTSemu(self,lim):
        for i in range(lim):
            bd=deepcopy(self.state)
            node=self.rn
            action=-1
            while True:
                if node.isleaf():
                    break
                action,node=node.select()
                bd.domove(action)
            actprobs,V=c.pubNN.predict(bd)
            if action==-1:
                winned=False
            else:
                winned,winner=bd.isWin(action)
            if winned:
                V=1 if winner==bd.player else -1
            elif bd.isDraw():
                V=0
            else:
                node.expand([(i,actprobs[0][i]) for i in bd.availablemoves]) 
            node.update(-V)

    def play(self,gm,silence=False):
        self.state=deepcopy(gm)
        if self.rn==None:
            self.rn=Leaf(None)
        elif not c.learning:
            newed=True
            fqp=deepcopy(self.state.qp)
            for a,i in self.rn.children.items():
                fqp[a]=self.state.player
                if fqp==gm.qp:
                    self.rn=i
                    newed=False
                    break
                fqp[a]=c.emptp
            if newed:
                self.rn=Leaf(None)
        if not silence: print("NolosZero正在思考中...")
        self.MCTSemu(c.NZtimes)
        if c.learning:
            probs=softmax(1.0/c.tau*np.log([i.N for a,i in self.rn.children.items()]))
            action=np.random.choice([a for a,i in self.rn.children.items()],p=c.epsilon*probs+(1-c.epsilon)*np.random.dirichlet(c.dlalpha*np.ones(len(probs))))
            self.rn=self.rn.children[action]
        else:
            probs=softmax(1000*np.log([i.N for a,i in self.rn.children.items()]))
            action=np.random.choice([a for a,i in self.rn.children.items()],p=probs)
            self.rn=self.rn.children[action]
        return action