
# coding: utf-8

# In[13]:

import numpy as np


# In[369]:

def DecisonTree(object):
    def __init(self,**kwargs):
        '''
        Height: the Max Height of Tree
        '''
        self.Height = Height
        self.task = task
        self.Tree = {}
        
    def train(self,train_X,train_y):
        '''
        对于类别属性来说，有两种处理方法：
        1. 按属性中所有的值来分类来生成树（C3.0等早期分类树就是这种方式）
        2. 将类别属性One-hot 编码来这样就只有0-1变量了。（Xgboost,CART 是这种)
        3. 其实如果按照最好的离散属性分割方式，应该穷举所有的组合，找到一个最好的分割方式。H20好像有这种实践。

        对于连续属性来说，就很好办了，直接二分分类。
        属性中的值来说，一般来说，如果是（0,1）变量，那么每次分类后就可以去掉这个特征了，对于连续变量就不是了，要继续保留参与分割。

        本算法采用方法2处理离散变量。即不再区分特征是离散变量还是连续变量，这样做能降低代码复杂度，而且做到了分类和回归任务的统一处理。
        本算法仅仅生成二叉树，不生成多叉树。
        '''
        '''
        不可采用分割方式，将子数据集传入递归函数，这样会丢失feature 位置，也就没法记录分割位置了，哈哈哈
        算了，还是切割吧。简化程序处理
        '''
        self.train_X = train_X
        self.train_y = train_y
        tree = TreeGenerate(train_X,train_y)
        self.Tree = pruning(tree)


# In[10]:

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
from sklearn.model_selection import train_test_split


# In[11]:

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)


# In[319]:

y_train[[1,2,3,7,99]]


# In[49]:

A = np.arange(X_train.shape[1])


# In[140]:

def chooseBestSplitfeature():
    return 3


# In[367]:

class self():
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    max_depth = 5
    min_samples_split = 4
    min_samples_leaf = 2
    min_impurity_decrease = 0
    
    


# In[576]:

def TreeGenerate(X_train,y_train):
    #X_train = self.X_train[indexes,cols]
    #y_train = self.y_train[indexes]
    best_feature,best_value = chooseBestSplitfeature(X_train,y_train)#选取最佳分裂属性，最佳分裂点
    print('best_feature,best_value::',best_feature,best_value)
    if best_feature is None:
        return best_value
    Tree = {}

    Tree['splitF'] = best_feature
    Tree['splitV'] = best_value
    left_X_train,left_y_train,right_X_train,right_y_train = binSplit(X_train,y_train,best_feature,best_value)
#     lef_cols = np.argwhere(np.logical_not((left_X_train==left_X_train[0]).all(axis=0))).flatten()
#     lef_cols = np.argwhere(np.logical_not((left_X_train==left_X_train[0]).all(axis=0))).flatten()
    Tree['left'] = TreeGenerate(left_X_train,left_y_train)
    Tree['right'] = TreeGenerate(right_X_train,right_y_train)
    return Tree
    
def chooseBestSplitfeature(X_train,y_train):
    cls,cnt = np.unique(y_train,return_counts=True)#统计y值的数量
    print('21,cls::',cls)
    if cls.shape[0]==1:#如果y值全部一样，停止分裂
        print('case 1:cls.shape[0]==1',cls)
        node = cls[0]
        return None,node
    if (X_train == X_train[0]).all():
        #如果属性值为空为空，或者X中全部样本相同，则返回y中出现最多的值
        print('case 2:== X_train[0]).all()', X_train[0])
        return None,get_NodeValue(y_train)
    if X_train.shape[0] < self.min_samples_split:# 如果X数量小于最小样本分割数，停止分裂
        print('case3 min_samples_split',X_train.shape[0])
        return None,get_NodeValue(y_train)
    
    current_error = cal_Error(y_train)
    print('current_error:',current_error)
    best_feature = 0
    best_value = get_NodeValue(y_train)
    best_error = cal_Error(y_train)
    for i in range(X_train.shape[1]):
        cls = np.unique(X_train[:,i])
        if cls.shape[0]==1:
            continue
        cls = np.delete(cls,np.argmin(cls))
        for val in cls:
            left_y_train,right_y_train = binSplit_y(X_train,y_train,i,val)
            if (left_y_train.shape[0] < self.min_samples_leaf) or            (right_y_train.shape[0] <self.min_samples_leaf):
                #print(left_y_train.shape,end=',')
                #print(right_y_train.shape,end=',')
                #error = cal_Error(y_train)
                pass
            else:
                error = cal_Error(left_y_train,right_y_train)
                if error < best_error:
                    #print('error:',error)
                    best_feature = i
                    best_value = val
                    best_error = error
        #print('\n')
    if current_error - best_error < self.min_impurity_decrease:
        print('case4 min_impurity_decrease',current_error - best_error)
        return None,get_NodeValue(y_train)
    
    return best_feature,best_value
            
            
def binSplit(X_train,y_train,feature,value):
    con_lt = np.where(X_train[:,feature]<value)
    con_get = np.where(X_train[:,feature]>= value)
    return X_train[con_lt],y_train[con_lt],X_train[con_get],y_train[con_get]            

def binSplit_y(X_train,y_train,feature,value):
    #print(66,feature,value,end=';')
    con_lt = np.where(X_train[:,feature]<value)
    con_get = np.where(X_train[:,feature]>= value)
    return y_train[con_lt],y_train[con_get]


def get_NodeValue(y_train):
    '''
    如果是分类任务，返回出现频率最高的一个值。
    如果是回归任务，返回数据平均值
    '''
    cls,cnt = np.unique(y_train,return_counts=True)
    return max(zip(cls,cnt),key=lambda x:x[1])[0]

def cal_Error(left_y_train,right_y_train=None):
    '''
    
    分类任务是 计算gini，熵；回归任务一般是计算MAE。
    refer:
    [1] scikit-learn. http://scikit-learn.org/stable/modules/tree.html#classification-criteria
    [2] 《机器学习》P79页4.2.3节基尼系数(周志华著作，西瓜书)
    '''
    #np.mean(y_train)
    if right_y_train is None:
        return get_gini(left_y_train)
    left_gini = get_gini(left_y_train)
    right_gini = get_gini(right_y_train)
    gini = np.array([left_gini,right_gini])
    shape_l = left_y_train.shape[0];shape_r = right_y_train.shape[0]
    weight = np.array([shape_l,shape_r])/(shape_l+shape_r)
    
    
    gini_index = (weight*gini).sum()
    
    return gini_index
    
def get_gini(y_train):
    
    lens = y_train.shape[0]
    vals,cnts = np.unique(y_train,return_counts=True)
    #print(vals,cnts,cnts/lens)
    gini = 1 - ((cnts/lens)**2).sum()# 就是这么简单...
    return gini
    
    

def pruning(Tree):
    '''
    如果你把生成的树的图化出来，你会发现有些树的非叶子节点左右子树的值一样，这样肯定是不合理的，所以要有一个剪枝的buzhou
    本函数只是为了优化树的结构，并不是用来防止过拟合的。
    '''
    if not isinstance(Tree,dict):
        return Tree
    if Tree['left'] == Tree['right']:
        return Tree['left']
    if isinstance(Tree['left'],dict):
        Tree['left'] = pruning(Tree['left'])
    if isinstance(Tree['right'],dict):
        Tree['right'] = pruning(Tree['right'])
    return Tree


# In[541]:

tree = TreeGenerate(X_train,y_train)


# In[556]:

tree['right']['left'] == tree['right']['right']


# In[574]:

tree2 = pruning(tree)


# In[575]:

tree2


# In[571]:

tree2['right']['right']['left']['right']


# In[572]:

'y' == 'y'


# In[532]:

tree['right']['right']['right']


# In[533]:

tree


# In[492]:

a=np.array([2,5])
b=np.array([5,10])


# In[509]:

a,b = binSplit_y(X_train,y_train,2,3)


# In[511]:

cal_Error(a,b)


# In[488]:

get_gini(y_train)


# In[473]:

A = np.array([1,1,2,2,3,3])

a,b = np.unique(A,return_counts=True)

1 - ((b/6)**2).sum()


# In[521]:

chooseBestSplitfeature(X_train,y_train)


# In[ ]:

X_train[:,2]


# In[491]:

np.unique(X_train[:,2])


# In[460]:

X_train[:,0]


# In[436]:

binSplit_y(X_train,y_train,0,4.4)


# In[430]:

con_lt = np.where(X_train[:,0]<5.9)
con_get = np.where(X_train[:,0]>= 5.9)
y_train[con_lt]
#y_train[con_get]


# In[431]:

y_train[con_get]


# In[432]:

np.unique(X_train[:,0])


# In[360]:

cls = np.delete(cls,np.argmin(cls))


# In[ ]:




# In[362]:

for i in cls:
    print(i)


# In[201]:

np.where(X_train[:,1]>3)[0].shape


# In[216]:

X_train.T.shape


# In[217]:

X_train.T==X_train.T[0]


# In[220]:

def myfunc(a,b):
    if (a>b): return a
    else: return b
vecfunc = np.vectorize(myfunc)
result=vecfunc([[1,2,3],[5,6,9]],[7,4,5])
print(result)


# In[229]:

X_train[np.unique(X_train)]


# In[238]:

A=np.array([[1,2,3,4],[1,3,3,2],[1,5,6,89]])


# In[239]:

A


# In[250]:

X_train


# In[271]:

A=np.tile(X_train,(50000,1))


# In[328]:

np.argwhere(np.logical_not((A==A[0]).all(axis=0)))


# In[337]:

np.argwhere(np.logical_not((A==A[0]).all(axis=0))).flatten()


# In[309]:

C=B


# In[310]:

C


# In[312]:

X_train[np.logical_not(C)]


# In[286]:

X_train


# In[300]:

C = B.all(axis=0)


# In[ ]:




# In[303]:

C


# In[304]:


np.argwhere([True,True,False,True])


# In[299]:

X_train[:,[True,True,False,True]]


# In[281]:

X_train.shape


# In[284]:

np.diff(X_train,axis=0)


# In[272]:

A[:,((A==A[0]).T).sum(axis=1)!=A.shape[1]]


# In[233]:

X_train


# In[231]:

X_train.T==X_train.T[0]


# In[221]:

vecfunc


# In[219]:

X_train==X_train[]


# In[214]:

np.count_nonzero(X_train[])


# In[211]:

np.unique(X_train[:,3])


# In[169]:

X_train[X_train[:,1]>]


# In[172]:

b = (X_train[:,1]>3)


# In[173]:

b.


# In[176]:

np.zeros([1,2,3])


# In[198]:

X_train[np.nonzero(X_train[:,1]>3)[0]].shape


# In[ ]:




# In[ ]:




# In[178]:

X_train[:,1]>3


# In[51]:

np.where()X_train


# In[58]:

(X_train[0] == X_train[0]).all()


# In[62]:

X_train[0].repeat(,axis=)


# In[65]:

(np.tile(X_train[0],(4,1))== X_train[0]).all()


# In[68]:

np.tile(X_train[0],(4,1))


# In[114]:

np.tile(X_train,(500,1)).shape


# In[112]:

(np.tile(X_train,(500,1))==X_train[0]).all()


# In[120]:

(np.diff(np.tile(X_train[0],(4,1)),1,0)==0).all()


# In[133]:

np.diff(np.tile(X_train[0],(4,1)),,0)==0


# In[81]:

X_train.shape

