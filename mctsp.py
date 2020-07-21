# -*- coding:utf-8;
#NolosZero的MCTS模拟

import consts as c
import numpy as np
from copy import deepcopy

class Leaf:
    def __init__(self,board,move,predP=1,parent=None):
        self.board=board
        self.parent=parent
        self.children=[]        
        self.N=0
        self.Q=0
        self.P=predP
        self.move=move
        self.gmend=False

    def isroot(self):
        return self.parent==None

    def isleaf(self):
        return len(self.children)<len(self.board.availablemoves) or self.gmend

    def update(self,val):
        self.Q=(self.Q*self.N+val)/float(self.N+1)
        self.N+=1
        if not self.isroot():
            self.parent.update(-val)
            
    def W(self):
        self.U=1.4*self.P*np.sqrt(self.parent.N)/(self.N+1)
        return self.Q+self.U

    def select(self):
        if self.isleaf():
            return self
        return max(self.children,key=lambda lnode:lnode.W()).select()

    def expand(self):
        if self.gmend:
            return self
        cb=deepcopy(self.board)
        fm=cb.availablemoves[len(self.children)]
        cb.domove(fm)
        cl=Leaf(cb,fm,1,self)
        winned,winner=cb.isWin(fm)
        if winned:
            cl.gmend=True
            cl.gmresult=1
        elif cb.isDraw():
            cl.gmend=True
            cl.gmresult=0
        self.children.append(cl)
        return cl

    def rollout(self):
        if self.gmend:
            self.update(self.gmresult)
            return
        tb=deepcopy(self.board)
        while True:
            fm=np.random.choice(tb.availablemoves)
            tb.domove(fm)
            winned,winner=tb.isWin(fm)
            if winned:
                self.update(-1 if winner==self.board.player else 1)
                return
            elif tb.isDraw():
                self.update(0)
                return

class MCTSPlayer:
    rn=None
    def MCTSemu(self,lim):
        for i in range(lim):
            self.rn.select().expand().rollout()

    def play(self,gm):
        if self.rn==None:
            self.rn=Leaf(deepcopy(gm),None)
        elif not c.learning:
            newed=True
            for i in self.rn.children:
                if i.board.qp==gm.qp:
                    self.rn=i
                    newed=False
                    break
            if newed:
                self.rn=Leaf(deepcopy(gm),None)
        print("电脑正在思考中...")
        self.MCTSemu(c.MCTStimes)
        '''for i in self.rn.children:
            print(str(i.move)+";"+str(i.N))'''
        self.rn=max(self.rn.children,key=lambda node:node.N)
        return self.rn.move