# -*- coding:utf-8;
# NolosZero的带有用户交互界面的人类玩家

import consts as c
from graphics import *


class HumanUIPlayer():
    def __init__(self):
        self.step=1
        self.window=None

    def move2loc(self, pos):
        return pos[0]*c.edge+pos[1]

    def loc2move(self, pos):
        return [pos // c.edge, pos % c.edge]

    def play(self, gm):
        if self.window==None:
            self.init_draw()
        if len(gm.availablemoves) > c.edge**2 - self.step + 1:
            self.window.close()
            self.step=1
            self.init_draw()
        if gm.lastmove != -1:
            self.render(self.loc2move(gm.lastmove), "black" if gm.player==c.player2 else "white")
        print("等待用户下棋...")
        pos=self.window.getMouse()
        pos=[pos.getY() // 34, pos.getX() // 34]
        self.render(pos, "white" if gm.player == c.player2 else "black")
        return self.move2loc(pos)

    def init_draw(self):
        self.window=GraphWin("与NolosZero进行五子棋对战", 34*c.edge, 34*c.edge)
        self.window.setBackground("lime")
        for i in range(c.edge):
            l=Line(Point(17+i*34,17), Point(17+i*34,17 + (c.edge-1)*34))
            l.draw(self.window)
            l=Line(Point(17,17+i*34), Point(17 + (c.edge-1)*34,17 + i*34))
            l.draw(self.window)

    def render(self,pos,color):
        location=Point(17+pos[1]*34,17+pos[0]*34)
        piece=Circle(location,15)
        piece.setFill(color)
        ptxt = Text(location, self.step)
        ptxt.setTextColor("white" if color=="black" else "black")
        piece.draw(self.window)
        ptxt.draw(self.window)
        self.step+=1