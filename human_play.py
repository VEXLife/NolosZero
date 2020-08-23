# -*- coding:utf-8;
# NolosZero的人类玩家

import consts as c


class HumanPlayer():
    def move2loc(self, pos):
        return pos[0]*c.edge+pos[1]

    def play(self, gm):
        fr = ""
        while fr == "":
            fr = input("请输入棋子坐标（%s方）：" % gm.player)
        return self.move2loc(eval(fr))
