# -*- coding:utf-8;
# NolosZero的游戏部分

import consts as c
from copy import deepcopy


class game:
    qp, availablemoves = [], []
    plane_e = [0 for i in range(c.edge**2)]

    def ps(self):
        print("本程序是基于AlphaZero算法的五子棋游戏")

    def draw(self):
        for i in range(c.edge):
            print("   "+str(i).rjust(2), end="")
        print()
        for i in range(c.edge):
            print(str(i).ljust(2) + str(self.qp[i*c.edge:i*c.edge+c.edge]))

    def newgame(self):
        self.player = c.player1
        self.qp, self.availablemoves = [c.emptp for i in range(
            c.edge**2)], [i for i in range(c.edge**2)]
        self.lastmove = -1
        self.plane = deepcopy(self.plane_e)

    def reverse(self):
        self.player = c.player1 if self.player == c.player2 else c.player2

    def isWin(self, i):
        lcount = 0
        chess = self.qp[i]
        for j in range(0, c.wn):
            # -
            v = i+j
            if v >= len(self.qp) \
                    or (v % c.edge == 0 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        for j in range(1, c.wn):
            # -
            v = i-j
            if v < 0 \
                    or (v % c.edge == c.edge-1 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        if lcount >= c.wn:
            return True, chess
        lcount = 0
        for j in range(0, c.wn):
            # |
            v = i+j*c.edge
            if v >= len(self.qp):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        for j in range(1, c.wn):
            # |
            v = i-j*c.edge
            if v < 0:
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        if lcount >= c.wn:
            return True, chess
        lcount = 0
        for j in range(0, c.wn):
            # /
            v = i+j*c.edge-j
            if v >= len(self.qp) \
                    or (v % c.edge == c.edge-1 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        for j in range(1, c.wn):
            # /
            v = i-j*c.edge+j
            if v < 0 \
                    or (v % c.edge == 0 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        if lcount >= c.wn:
            return True, chess
        lcount = 0
        for j in range(0, c.wn):
            # \
            v = i+j*c.edge+j
            if v >= len(self.qp) \
                    or (v % c.edge == 0 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        for j in range(1, c.wn):
            # \
            v = i-j*c.edge-j
            if v >= len(self.qp) or v < 0 \
                    or (v % c.edge == c.edge-1 and j > 0):
                break
            if self.qp[v] == chess:
                lcount += 1
            else:
                break
        if lcount >= c.wn:
            return True, chess
        return False, c.emptp

    def isDraw(self):
        return self.availablemoves == []

    def domove(self, pos):
        self.qp[pos] = self.player
        if self.player == c.player1:
            self.plane[pos] = 1
        else:
            self.plane[pos] = -1
        self.availablemoves.remove(pos)
        self.lastmove = pos
        self.reverse()

    def start(self):
        self.newgame()
        while True:
            self.draw()
            p = c.player1player.play(self)
            self.domove(p)
            winned, winner = self.isWin(p)
            if(winned == True):
                self.draw()
                print(str(winner)+"方获胜！")
                self.newgame()
                continue
            if self.isDraw():
                self.draw()
                print("平局！")
                self.newgame()
                continue
            self.draw()
            p = c.player2player.play(self)
            self.domove(p)
            winned, winner = self.isWin(p)
            if(winned == True):
                self.draw()
                print(str(winner)+"方获胜！")
                self.newgame()
                continue
            if self.isDraw():
                self.draw()
                print("平局！")
                self.newgame()
                continue
