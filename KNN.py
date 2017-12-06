
# coding: utf-8

# In[3]:

import numpy as np


# In[24]:

class Knn():
    def __init__(self,**kwargs):
        '''
        topK : Use top K nearest train samples 's label. to get the most frequency label.
        '''
        self.topK = kwargs.get('topK')
        
    def train(self,train_X,train_y):
        '''
        Knn 没有显式的训练过程
        '''
        self.train_X = train_X
        self.train_y = train_y
        self.len_train = self.train_X.shape[0]
        
        
    def predict(self,test_X):
        self.len_test = test_X.shape[0]
        train_X2 = np.tile(self.train_X,(self.len_test,1))
        test_X2 = np.repeat(test_X,self.len_train,axis=0)
        C1 = test_X2-train_X2
        D1 = (C1**2).sum(axis=1)**0.5
        D2 = D1.reshape((self.len_test,self.len_train))
        D3 = D2.argsort(axis=1)[:,:self.topK]
        D4 = self.train_y[D3]
        return np.array([max(zip(*np.unique(x,return_counts=True)),key=lambda x:x[1])[0] for x in D4])
        
        
        
        
        
        

k = Knn(topK=4)


# -------------------------------------------------------
# - #### 9行代码就实现了Knn算法当然是最简版的，这个算法是我手写的，
# - #### 中间经历了很多次的重构，刚开始代码非常臃肿，并且有一个对Numpy 的错误认识导致的错误
# - Numpy做数值计算真的不错不错，
# - 本算法只能用来做数值型距离计算，如果是非数值距离（如MDV距离)那么还要其他地方来实现，其实可以单独做一个功能，因为在聚类算法中也要有这样的计算
# - 我的算法非常简洁，我觉得应该是很简洁了

# ## 测试1

# In[11]:

if __name__ == '__main__':
    train_X = np.array([[1,2,3,5],[5,6,7,1],[9,4,-7,5],[5,6,44,3],[93,4,-7,51]])
    train_y = np.array(['Y','N','Y','N','N'])
    test_X = np.array([[3,5,6,7],[3,90,2,6]])

    k.train(train_X,train_y)
    k.predict(test_X)


# ## 测试2

# In[10]:

if __name__ == '__main__':

    k = Knn(topK=1)
    k.train(np.array([[0,0],[1,1],[9,10]]),np.array(['A','A','B']))
    k.predict(np.array([[0,1],[9,9],[3,3]]))


# ## 测试3

# In[30]:

if __name__ == '__main__':
    Int = {'largeDoses':3,'smallDoses':2,'didntLike':1}
    def file2matrix(filename):
        fr = open(filename)
        numberOfLines = len(fr.readlines())         #get the number of lines in the file
        returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
        classLabelVector = []                       #prepare labels return   
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(Int[listFromLine[-1]])
            index += 1
        return returnMat,classLabelVector

    def autoNorm(dataSet):
        minvals = dataSet.min(0)
        maxvals = dataSet.max(0)
        ranges = maxvals - minvals
        normDataSet = np.zeros(np.shape(dataSet))
        m =  dataSet.shape[0]
        normDataSet = dataSet - np.tile(minvals,(m,1))
        normDataSet = normDataSet/np.tile(ranges, (m,1))
        return normDataSet , ranges,minvals
    datingDatMat,datingLabels = file2matrix('../datingTestSet.txt')
    normMat,ranges ,minvals = autoNorm(datingDatMat)

    k = Knn(topK=4)
    k.train(normMat[300:],np.array(datingLabels[300:]))
    ret = k.predict(normMat[:300])


# In[ ]:



