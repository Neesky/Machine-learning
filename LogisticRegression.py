# encoding: utf-8
# @Time : 2021/10/23 14:05
# @Author : Neesky
# @contact: neesky@foxmail.com
# @File : LogisticRegression.py
# @Software: PyCharm

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

class LogisticRegressionMP(object):

    def __init__(self, lr=0.1, max_iter=3000, seed=None, weight_decay=True,minlr = 0.0001):
        '''
        :param lr:学习率，默认0.1
        :param max_iter:最大迭代数，默认3000
        :param seed: 随机种子，默认随时间变换
        :param weight_decay: 学习率是否衰减，默认True
        :param minlr: 最小学习率，默认0.0001
        '''
        if(seed != None):
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time()))
        self.eps = 1e-8
        self.lr = lr
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.decalylr = (lr-minlr)/max_iter
    def load_data(self,path,shuffle=True):
        '''
        :param path:数据路径(支持xls和csv读入)
        :param shuffle:是否打乱数据，默认True
        :return: 训练数据和标签
        '''
        types = {}
        if(path.endswith("csv")) :
            import csv
            import csv
            from itertools import islice
            X = []
            y = []
            with open(path, 'r') as read_file:
                reader = csv.reader(read_file)
                for row in islice(reader, 1, None):
                    X.append([eval(num) for num in row[1:-1]])
                    if (row[-1] not in types.keys()):
                        types[row[-1]] = types.__len__()
                    y.append(types[row[-1]])
            X,y = np.array(X),np.array(y)
        else :
            import xlrd
            data = xlrd.open_workbook(path)
            table = data.sheets()[0]
            nrows = table.nrows
            ncols = table.ncols
            X = np.zeros([nrows - 1, ncols - 2], dtype=np.float)
            y = np.zeros([nrows - 1], dtype=np.int)
            for i in range(1, nrows): #忽略第一排
                for j in range(1, ncols - 1): #忽略序号和标签
                    X[i - 1][j - 1] = table.cell(i, j).value
                if(table.cell(i, ncols - 1).value not in types.keys()):
                    types[table.cell(i, ncols - 1).value] = types.__len__()
                y[i - 1] = types[table.cell(i, ncols - 1).value]
        if shuffle :
            permutation = np.random.permutation(y.shape[0])
            X = X[permutation, :]
            y = y[permutation]
        return X,y
    def _cal_sigmod(self,lineResult):
        return 1/(1+np.exp(-1*lineResult))
    def _cal_y_predict(self,X):
        return self._cal_sigmod(np.dot(X,self.w) + self.b)
    def _cal_loss(self,y,y_):
        return np.mean(-1*(y*np.log2(y_ + self.eps)+(1-y)*np.log2(1-y_ + self.eps)))
    def _calc_gradient(self,y,y_,X):
        grand_w = np.dot((y_ - y),X) / len(y)
        grand_b = np.mean(y_ - y)
        return grand_w,grand_b
    def _updata_var(self,y,y_,X):
        grand_w, grand_b = self._calc_gradient(y,y_,X)
        self.w = self.w - self.lr * grand_w
        self.b = self.b - self.lr * grand_b
    def _cal_acc(self, y , y_ ):
        acc = np.mean([1 if ((y_[i]>=0.5 and y[i]==1) or (y_[i]<0.5 and y[i]==0)) else 0 for i in range(len(y))])
        return acc
    def splitData(self,X,y,ratio=0.7):
        '''
        :param X:数据集数据
        :param y:数据集标签
        :param ratio: 训练集：测试机 = ratio
        :return:训练集和验证集
        '''
        num = int(len(y)*ratio)
        X_train = X[0:num]
        y_train = y[0:num]
        X_test = X[num:]
        y_test = y[num:]
        return X_train,y_train,X_test,y_test
    def train(self , X , y ,interval = 50,draw = ["loss","acc"]):
        '''
        :param X:训练集数据
        :param y:训练集标签
        :param interval: 多少个epoch输出一次当前loss和acc，默认50
        :param draw: 画图选择，可选为"loss"和"acc",默认都选
        :return:
        '''
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)

        lossdraw = []
        accdraw = []

        for i in range(self.max_iter):

            y_ = self._cal_y_predict(X)
            # print(y_.shape,y.shape,X.shape,self.w.shape,self.b.shape)
            loss = self._cal_loss(y,y_)
            # print(y_.shape, y.shape, X.shape,self.w.shape,self.b.shape)
            acc = self._cal_acc(y,y_)
            # print(y_.shape, y.shape, X.shape,self.w.shape,self.b.shape)
            self._updata_var(y,y_,X)
            # print(y_.shape, y.shape, X.shape,self.w.shape,self.b.shape)
            if(i%interval != interval-1):
                print("Epoch[{}][{}] loss:{}".format(i,self.max_iter,loss))
            else :
                print("Epoch[{}][{}] loss:{} acc:{}".format(i,self.max_iter,loss,acc))
            if(self.weight_decay):
                self.lr = self.lr-self.decalylr
            lossdraw.append(loss)
            accdraw.append(acc)
        legend = []

        for drawstr in draw:
            if(drawstr+"draw" in locals().keys()):
                plt.plot(range(self.max_iter), locals()[drawstr+"draw"])
                legend.append(drawstr)
            else:
                print("<"+drawstr+">"+"没有被记录,请换一个指标(acc,loss)")
        if(len(legend)>0):
            plt.legend(legend)
            plt.show()
    def predict(self, X, y):
        '''
        :param X: 测试集数据
        :param y: 测试集标签
        :return:
        '''
        y_ = self._cal_y_predict(X)
        acc = self._cal_acc(y, y_)
        print("测试集最终acc:{}".format(acc))
if __name__ == "__main__":
    LR = LogisticRegressionMP(lr=0.1,max_iter=3000,weight_decay = True)
    X,y = LR.load_data(r'./Watermelon_data.xls',shuffle = True)
    X_train,y_train,X_test,y_test = LR.splitData(X,y)
    LR.train(X_train,y_train,draw = ["loss","acc"])
    LR.predict(X_test,y_test)