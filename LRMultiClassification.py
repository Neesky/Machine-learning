# encoding: utf-8
# @Time : 2021/10/25 23:14
# @Author : Neesky
# @contact: neesky@foxmail.com
# @File : LogisticRegression.py
# @Software: PyCharm
from LogisticRegression import LogisticRegressionMP
import numpy as np
class LRMultiClassificationMP(LogisticRegressionMP):
    def __init__(self):
        pass
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
            X,y = np.array(X).astype(np.int),np.array(y).astype(np.float)
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
    def processData0_no0(self,X,y):
        '''
        :param X:训练集数据
        :param y:训练集标签
        :return: 第一层分类器（种类0和非种类0的训练集）
        '''
        X_re = []
        y_re = []
        for i in range(len(X)):
            X_re.append(X[i])
            if(y[i] != 0):
                y_re.append(1)
            else :
                y_re.append(0)
        return np.array(X_re).astype(np.int),np.array(y_re).astype(np.float)
    def processData1_2(self,X,y):
        '''
        :param X:训练集数据
        :param y:训练集标签
        :return: 第二层分类器（种类1和种类2的训练集）
        '''
        X_re = []
        y_re = []
        for i in range(len(X)):

            if (y[i] == 1):
                X_re.append(X[i])
                y_re.append(0)
            elif y[i] == 2:
                X_re.append(X[i])
                y_re.append(1)
        return np.array(X_re).astype(np.int), np.array(y_re).astype(np.float)
    def predict(self,LR1,LR2,X, y):
        '''
        :param LR1:训练好的第一层分类器
        :param LR2:训练好的第二层分类器
        :param X: 测试集标签
        :param y: 测试集标签
        :return:
        '''

        y_ = np.zeros(len(y),dtype=np.int)

        y_predict1 = LR1._cal_y_predict(X)

        index2 = []
        X_train2 = []

        for i in range(len(y_predict1)):
            if(y_predict1[i]<0.5):
                y_[i] = 0
            else :
                index2.append(i)
                X_train2.append(X[i])

        y_predict2 = LR2._cal_y_predict(np.array(X_train2).astype(np.float))
        for i in range(len(y_predict2)):
            if (y_predict2[i] < 0.5):
                y_[index2[i]] = 1
            else:
                y_[index2[i]] = 2
        acc = np.mean(y==y_)
        print("测试集最终acc:{}".format(acc))
if __name__ == "__main__":
    LRM = LRMultiClassificationMP()
    X,y = LRM.load_data(r'./iris.csv',shuffle = True)
    X_train,y_train,X_test,y_test = LRM.splitData(X,y)
    X_train1,y_train1 = LRM.processData0_no0(X,y)
    X_train2,y_train2 = LRM.processData1_2(X,y)

    LR1 = LogisticRegressionMP(lr=0.01,max_iter=3000,weight_decay = True)
    LR1.train(X_train1,y_train1)
    LR2 = LogisticRegressionMP(lr=0.01,max_iter=3000,weight_decay = True)
    LR2.train(X_train2,y_train2)

    LRM.predict(LR1,LR2,X_test,y_test)

