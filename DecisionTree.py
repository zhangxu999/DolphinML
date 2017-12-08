
# coding: utf-8

# In[123]:

import numpy as np

class DecisonTree(object):
    def __init__(self,**kwargs):
        '''
        按照CART实现
        task:Classification | Regression
        or 后 都是 不传参数的默认值
        '''
        self.task = kwargs.get('task') or 'Classification'
        self.max_depth =  kwargs.get('max_depth') or np.infty
        min_samples_split = kwargs.get('min_samples_split')
        self.min_samples_split = kwargs.get('min_samples_split') or 2
        self.min_samples_leaf = kwargs.get('min_samples_leaf') or 1
        self.min_impurity_decrease = kwargs.get('min_samples_leaf') or 0

        
    def train(self,X_train,y_train):
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
        self.X_train = X_train
        self.y_train = y_train
        tree = self.TreeGenerate(X_train,y_train)
        self.Tree = self.pruning(tree)
    def predict(self,X_test):
        def get_node_value(X):
            T = self.Tree
            while True:
                F = T['splitF']
                V = T['splitV']
                T = T['left'] if X[F] < V else T['right'] 
                if not isinstance(T,dict):
                    return T
        ret = np.apply_along_axis(get_node_value,1,X_test)
        return ret


    def TreeGenerate(self,X_train,y_train,depth = 0):
        best_feature,best_value,best_error = self.chooseBestSplitfeature(X_train,y_train,depth)#选取最佳分裂属性，最佳分裂点
        #print('best_feature,best_value::',best_feature,best_value)
        if best_feature is None:#如果best_feature 为空，说明当前节点不满足分裂条件，函数直接返回节点值。
            return best_value
        Tree = {}
        Tree['splitF'] = best_feature
        Tree['splitV'] = best_value
        Tree['best_error'] = best_error
        Tree['depth'] = depth
        left_X_train,left_y_train,right_X_train,right_y_train = self.binSplit(X_train,y_train,best_feature,best_value)
        Tree['left'] = self.TreeGenerate(left_X_train,left_y_train,depth+1)
        Tree['right'] = self.TreeGenerate(right_X_train,right_y_train,depth+1)
        return Tree

    def chooseBestSplitfeature(self,X_train,y_train,depth):
        cls,cnt = np.unique(y_train,return_counts=True)#统计y值的数量
        Error_ytrain = self.cal_Error(y_train)
        NodeV_ytrain = self.get_NodeValue(y_train)
        if (cls.shape[0]==1) or (depth > self.max_depth) or         ((X_train == X_train[0]).all()) or (X_train.shape[0] < self.min_samples_split):
            #如果y值全部一样，停止分裂;超过最大高度，停止分裂;X中全部样本相同，停止分裂;X数量小于最小样本分割数，停止分裂
            return None,NodeV_ytrain,Error_ytrain
#         if depth > self.max_depth:#超过最大高度，停止分裂
#             return None,get_NodeValue(y_train),Error_ytrain
#         if (X_train == X_train[0]).all():
#             #如果属性值为空为空，或者X中全部样本相同，则返回y中出现最多的值
#             return None,get_NodeValue(y_train),Error_ytrain
#         if X_train.shape[0] < self.min_samples_split:# 如果X数量小于最小样本分割数，停止分裂
#             return None,get_NodeValue(y_train),Error_ytrain

        current_error = Error_ytrain
        #print('current_error:',current_error)
        best_feature = 0
        best_value = NodeV_ytrain
        best_error = Error_ytrain
        for i in range(X_train.shape[1]):
            cls = np.unique(X_train[:,i])
            if cls.shape[0]==1:
                continue
            cls = np.delete(cls,np.argmin(cls))#去掉一个最小值，开始枚举
            for val in cls:
                left_y_train,right_y_train = self.binSplit_y(X_train,y_train,i,val)
                if (left_y_train.shape[0] < self.min_samples_leaf) or                (right_y_train.shape[0] <self.min_samples_leaf):#如果分裂后样本数量小于min_samples_leaf，停止分裂
                    pass
                else:#求较小的error
                    error = self.cal_Error(left_y_train,right_y_train)
                    if error < best_error:
                        best_feature = i
                        best_value = val
                        best_error = error
        if current_error - best_error <= self.min_impurity_decrease:
            #Error减少过少，停止分裂
            #print('case4 min_impurity_decrease',current_error - best_error)
            return None,Error_ytrain,current_error

        return best_feature,best_value,best_error

    def binSplit(self,X_train,y_train,feature,value):
        con_lt = np.where(X_train[:,feature]<value)
        con_get = np.where(X_train[:,feature]>= value)
        return X_train[con_lt],y_train[con_lt],X_train[con_get],y_train[con_get]            

    def binSplit_y(self,X_train,y_train,feature,value):
        #print(66,feature,value,end=';')
        con_lt = np.where(X_train[:,feature]<value)
        con_get = np.where(X_train[:,feature]>= value)
        return y_train[con_lt],y_train[con_get]

    def get_NodeValue(self,y_train):
        '''
        如果是分类任务，返回出现频率最高的一个值。
        如果是回归任务，返回数据平均值
        '''
        if self.task == 'Classification':
            cls,cnt = np.unique(y_train,return_counts=True)
            return max(zip(cls,cnt),key=lambda x:x[1])[0]
        elif self.task == 'Regression':
            return np.mean(y_train)

    def cal_Error(self,left_y_train,right_y_train=None):
        '''

        分类任务是 计算gini，熵,也可以用其他评估函数，比如交叉熵；回归任务一般是计算MAE。
        refer:
        [1] scikit-learn. http://scikit-learn.org/stable/modules/tree.html#classification-criteria
        
        '''
        #np.mean(y_train)
        if right_y_train is None:
            return self.get_gini(left_y_train)
        
        left_gini = self.get_gini(left_y_train)
        right_gini = self.get_gini(right_y_train)
        gini = np.array([left_gini,right_gini])
        shape_l = left_y_train.shape[0];shape_r = right_y_train.shape[0]
        weight = np.array([shape_l,shape_r])/(shape_l+shape_r)
        gini_index = (weight*gini).sum()
        return gini_index

    def get_gini(self,y_train):
        '''
        计算基尼系数，分类任务中会用到。
        refer:
        [1] 《机器学习》P79页4.2.3节基尼系数(周志华著，西瓜书)
        '''
        lens = y_train.shape[0]
        vals,cnts = np.unique(y_train,return_counts=True)
        #print(vals,cnts,cnts/lens)
        gini = 1 - ((cnts/lens)**2).sum()# 就是这么简单...
        return gini

    def pruning(self,Tree):
        '''
        如果你把生成的树的图化出来，你会发现有些树的非叶子节点左右子树的值一样，这样肯定是不合理的，所以要有一个剪枝的buzhou
        本函数只是为了优化树的结构，并不是用来防止过拟合的。
        '''
        if not isinstance(Tree,dict):
            return Tree
        if Tree['left'] == Tree['right']:
            return Tree['left']
        if isinstance(Tree['left'],dict):
            Tree['left'] = self.pruning(Tree['left'])
        if isinstance(Tree['right'],dict):
            Tree['right'] = self.pruning(Tree['right'])
        return Tree


# ## 测试一

# In[122]:

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn import tree
    iris = load_iris()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    DT = DecisonTree()
    DT.train(X_train,y_train)
    p = DT.predict(X_test)


# ## 一段利于调试编写的工具代码

# In[118]:

if __name__ == '__main__':
    class self():
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
        max_depth = 5 
        min_samples_split = 4
        min_samples_leaf = 2
        min_impurity_decrease = 0
        max_depth = 3
    
    

