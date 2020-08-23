# -*- coding:utf-8;
# NolosZero的参数汇总

from human_play import HumanPlayer
from mctsp import *  # MCTSPlayer
from pvn import CNN
from mctsn import *  # NolosZeroPlayer

edge = 5  # 棋盘边长
player1 = "X"  # 先手的棋
player2 = "O"  # 后手的棋
player1player = NolosZeroPlayer()  # 先手玩家
player2player = MCTSPlayer()  # 后手玩家
emptp = "_"  # 空棋格
wn = 4  # 赢子数量
l2c = 1e-4  # L2正则化优化器常数
lr = 1e-3  # 初始学习率
filters = {"head": [64], "residual": [[64, 64]], "policy": [64]}  # 过滤器数量
model_file_name = "4_on_5_5.nolosknowledge"  # 模型文件夹名
c_puct = 1.25  # 探索系数
c_base = 19652  # 探索基数
tau = 10.0  # 温度系数
epsilon = 0.75  # 探索程度
MCTStimes = 1000  # 蒙特卡洛模拟次数
NZtimes = 400  # NolosZero模拟次数
learning = False  # 是否在训练中
epochs = 5  # 训练轮数
minbatch = 256  # 最小批数据大小
train_data_file_name = "train_data_" + model_file_name  # 训练数据缓存文件名
log_file_name = model_file_name + ".log"  # 日志文件文件名
model_plot_file_name = model_file_name + ".png"  # 神经网络概览图文件名
buffer_size = 512  # 训练数据缓存器最大空间
gamecount = 0  # 游戏计数
pubNN = CNN()  # 人工神经网络对象
