import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings

class LogisticRegressionMP(object):

    def __init__(self, lr=0.1, max_iter=100, seed=None, weight_decay=True,minlr = 0.0001):
        if(seed != None):
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time()))
        self.lr = lr
        self.max_iter = max_iter
        self.weight_decay = weight_decay
        self.decalylr = (lr-minlr)/max_iter
    def load_data(self,path,shuffle=True):
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
            y[i - 1] = 1 if table.cell(i, ncols - 1).value == "是" else 0
        if shuffle :
            permutation = np.random.permutation(y.shape[0])
            X = X[permutation, :]
            y = y[permutation]
        return X,y
    def cal_sigmod(self,lineResult):
        return 1/(1+np.exp(-1*lineResult))
    def cal_y_predict(self,X):
        return self.cal_sigmod(np.dot(X,self.w) + self.b)
    def cal_loss(self,y,y_):
        return np.mean(-1*(y*np.log2(y_)+(1-y)*np.log2(1-y_)))
    def calc_gradient(self,y,y_,X):
        grand_w = np.dot((y_ - y),X) / len(y)
        grand_b = np.mean(y_ - y)
        return grand_w,grand_b
    def updata_var(self,y,y_,X):
        grand_w, grand_b = self.calc_gradient(y,y_,X)
        self.w = self.w - self.lr * grand_w
        self.b = self.b - self.lr * grand_b
    def cal_acc(self, y , y_ ):
        acc = np.mean([1 if ((y_[i]>=0.5 and y[i]==1) or (y_[i]<0.5 and y[i]==0)) else 0 for i in range(len(y))])
        return acc
    def splitData(self,X,y,ratio=0.7):
        num = int(len(y)*ratio)
        X_train = X[0:num]
        y_train = y[0:num]
        X_test = X[num:]
        y_test = y[num:]
        return X_train,y_train,X_test,y_test
    def train(self , X , y ,interval = 50,draw = ["lossdraw","accdraw"]):
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand(1)

        lossdraw = []
        accdraw = []

        for i in range(self.max_iter):

            y_ = self.cal_y_predict(X)
            # print(y_.shape,y.shape,X.shape,self.w.shape,self.b.shape)
            loss = self.cal_loss(y,y_)
            # print(y_.shape, y.shape, X.shape,self.w.shape,self.b.shape)
            acc = self.cal_acc(y,y_)
            # print(y_.shape, y.shape, X.shape,self.w.shape,self.b.shape)
            self.updata_var(y,y_,X)
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
        y_ = self.cal_y_predict(X)
        acc = self.cal_acc(y, y_)
        print("acc:{}".format(acc))
if __name__ == "__main__":
    LR = LogisticRegressionMP(lr=0.1,max_iter=3000,weight_decay = True)
    X,y = LR.load_data(r'./Watermelon_data.xls',shuffle = True)
    X_train,y_train,X_test,y_test = LR.splitData(X,y)
    LR.train(X_train,y_train,draw = ["loss","acc"])
    LR.predict(X_test,y_test)