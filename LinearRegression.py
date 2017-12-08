
# coding: utf-8

# In[73]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[61]:

class LogisticRegression():
    def __init__(self,**kwargs):
        self.max_iter = kwargs.get('max_iter') or 1000 # 梯度下降最大迭代次数
        self.tol = kwargs.get('tol') or 1e-4 # 梯度下降最小 误差减小值（未使用
        self.learning_rate = kwargs.get('learning_rate') or 0.01 # 下降速度
        self.task = kwargs.get('task') or 'Classification' #任务类型
        self.use_sigmoid = kwargs.get('use_sigmoid') or False #是否使用sigmoid 函数拟合
        
    def train(self,X_train,y_train):
        '''
        使用梯度下降进行优化
        '''
        self.X_train = np.append(X_train,np.ones((X_train.shape[0],1)),1) # 补全常数项的值
        self.y_train,self.y_map = self.get_y_value(y_train) #处理y值，map 0,1 与真实值对应关系
        self.weights,self.error = self.Gradient_Descent(self.X_train,self.y_train) #梯度下降优化系数
        self.coef_ = self.weights[:-1] #系数
        self.intercept_ = self.weights[-1] #截距
    def predict(self,X_test):
        '''
        预测，矩阵相乘。分类问题结果匹配
        '''
        estimate_value = X_test*self.coef_+self.intercept_
        if self.task == 'Classification':
            sigmoid_value = self.sigmoid(estimate_value)
            y_one_ = self.y_map.get(1)
            return_value = np.full(sigmoid_value.shape,y_one_,dtype=type(y_one_))
            return_value[np.argwhere(sigmoid_value<0.5)] = self.y_map.get(0)
            return return_value
        else:
            return estimate_value


    def Gradient_Descent(self,X_train,y_train):
        labelMat = np.mat(y_train).T #convert to NumPy matrix
        m,n = X_train.shape
        weights = np.ones((n,1)) # 权重初始值设为1 ，也可设为别的值。
        for i in range(self.max_iter):
            h = self.get_estimate_value(X_train*weights) # 转化estimate值
            error = h - y_train# 计算残差
            #box.append(np.abs(error).sum())# 如过下降过慢就停止梯度下降
            #if (np.abs(error).sum() - pre_error) < self.tol:
                #return weights
            weights = weights - self.learning_rate*(X_train.T*error) 
            #更新权重，这一句对于初学者来说比较难以理解。请对照图形想象理解。
        return weights,np.abs(error).sum()
    def get_estimate_value(self,raw_estimate):
        '''
        对估计值转化
        '''
        if self.task == 'Classification':
            return self.sigmoid(raw_estimate)
        elif self.task == 'Regression':
            if self.use_sigmoid:
                return self.sigmoid(raw_estimate)
            else:
                return raw_estimate
    def get_y_value(self,raw_y_train):
        '''
        对y值转化
        '''
        if self.task == 'Classification':
            cls = np.unique(np.array(raw_y_train))[:2]
            new_ytrain = np.ones(raw_y_train.shape)
            for i in range(cls.shape[0]):
                new_ytrain[np.argwhere(raw_y_train==cls[i])] = i
            return new_ytrain,dict(zip([0,1],cls))
        else:
            if self.use_sigmoid:
                return self.sigmoid(raw_y_train),None
            else:
                return raw_y_train,None
            
    def sigmoid(self,X):
        '''
        S型函数
        '''
        return  1.0/(1+np.exp(-X))    


# In[70]:

if __name__ == '__main__':
    def unit_step(x):
        '''
        阶跃函数，将此函数替换sigmoid函数，你会发现，函数怎么也不收敛。
        '''
        if x<0:
            return 0
        elif x>0:
            return 1
        else:
            return 0.5
    def unsigmoid(x):
        '''
        sigmoid函数的逆过程
        '''
        return -np.log(1/x -1)


# In[71]:

if __name__ == '__main__':
    def loadDataSet():
        dataMat = []; labelMat = []
        fr = open('../Ch05/testSet.txt')
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat,labelMat

    a,b = loadDataSet()
    a = np.mat(a)
    b = np.mat(b).T
    LR = LogisticRegression(use_sigmoid=True,learning_rate=0.01,max_iter=1000)
    LR.train(a[:,1:],b)
    LR.predict(a[:,1:])


# In[83]:

if __name__ == '__main__':
    '''
    梯度下降错误变化图
    '''
    box = []
    def Gradient_Descent(X_train,y_train):
            labelMat = np.mat(y_train).T #convert to NumPy matrix
            m,n = X_train.shape
            weights = np.ones((n,1)) # 权重初始值设为1 ，也可设为别的值。最后的权重值会变化
            for i in range(500):
                h = LogisticRegression.get_estimate_value('s',X_train*weights) # 转化estimate值
                error = h - y_train# 计算残差
                #box.append(np.abs(error).sum())# 如过下降过慢就停止梯度下降
                #if (np.abs(error).sum() - pre_error) < self.tol:
                    #return weights
                weights = weights - 0.001*(X_train.T*error) 

    Gradient_Descent(a,b)
    plt.figure(figsize=(16,9))
    plt.plot(range(len(box)),box)
    box[-5:]

