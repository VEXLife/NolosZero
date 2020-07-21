# -*- coding:utf-8;
#NolosZero的参数汇总

from human_play import HumanPlayer
from mctsp import *#MCTSPlayer
from pvn import CNN
from mctsn import *

edge=8#棋盘边长
player1="X"#先手的棋
player2="O"#后手的棋
player1player=NolosZeroPlayer()#先手玩家
player2player=HumanPlayer()#后手玩家
emptp="_"#空棋格
wn=5#赢子数量
filters=(32,64,128,4,2)#神经网络卷积层过滤器数量
planecount=4#神经网络描述平面数量
FCs=(64,1)#神经网络全连接层神经元数量
l2c=1e-4#L2正则化优化器常数
model_file_name="5_on_8_8.nolosknowledge"#模型文件名
c_puct=5#探索系数
tau=1.0#温度系数
MCTStimes=1000#蒙特卡洛模拟次数
NZtimes=400#NolosZero模拟次数
learning=False#是否在训练中
epochs=5#训练轮数
epsilon=0.75#狄利克雷随机程度系数
dlalpha=0.3#狄利克雷随机参数
minbatch=512#最小批数据大小
bnaxis=1#批规范化轴
momentum=0.9#批规范化动量
train_data_file_name="train_data_" + model_file_name#训练数据缓存文件名
log_file_name=model_file_name + ".log"#日志文件文件名
factor=1.0#学习率乘子
min_delta=0.02#允许的最小损失变化量
lr=2e-3#初始学习率
min_lr=0.0002#最小学习率
buffer_size=10000#训练数据缓存器最大空间
gamecount=0#游戏计数
pubNN=CNN()#人工神经网络对象