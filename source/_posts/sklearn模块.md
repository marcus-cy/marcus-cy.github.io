---
title: python之旅
date: 2018-12-13 15:28:52
categories: 写作4
---

# Numpy模块
### array

```
np.array() 创建N维数组对象，元素必须相同类型，每个数组都有一个shape和一个dyte
**和列表最重要的区别，数组切片是原始数组的视图，即视图上的任何修改都会直接反映到源数组上，如果需要的是一份副本而非视图，即操作arr[:].copy()**


常用属性：
np.nan；-np.inf；np.inf；
np.arange()；np.ones()；np.mat() ；np.zeros((2,2))；np.eyes(4)
np.reshape(-1,1) #转化为一列，-1代表自动计算行数


常用合并：
np.vstack() #对array进行上下合并
np.hstack() #对array进行横向合并
np.r_[a,b,c] #类似pandas中的concat
np.c_[a,b,c] #类似pandas中的merge
np.column_stack()# 类似hstack，将每个元素作为一列
np.concatenate([a,b],axis=1) # 对array进行合并,同hstack()


np.sort(array, axis, kind, order) # 返回新数组
np.diff() #数组相邻两个元素之间的差
np.split() # 如果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置
np.repeat([,axis]) #将数组中的各个元素重复一定次数
np.tile() #堆叠数组副本
a[:, np.newaxis] # 给a最外层中括号中的每一个元素加[]
a[newaxis, :] # 给a最外层中括号中所有元素加[]




随机取数：
np.linspace(0,10,50)
#返回50个均匀分布的样本，在[0, 10]之间

numpy.random.randn(d0, d1, ..., dn)
#是从标准正态分布中返回一个或多个样本值，dn指维数

numpy.random.rand(d0, d1, …, dn)
#随机样本位于[0, 1)中 ，dn指维数

numpy.random.randint(low,high=None,size)
#生成在[low,high) 之间均匀分布的样本，若high为none，区间为[0，low)

np.random.normal(mean,stdev,size)
#均值为mean，标准差为stdev，返回size个高斯随机数

numpy.random.choice(a, size=None, replace=True, p=None)
从a中随机选取size个数量，replace为True时，采样会重复

```

### 集合运算

```
intersect1d(x,y) 返回x、y公共元素
union1d(x,y)  返回x、y并集
in1d(x,y)  返回x元素是否在y中
setdiff1d(x,y) 集合差，含于x，不含y
setxor1d(x,y) x、y中非并集的元素
```

### 索引运算
```
names[names=='a']
data=np.random.randn(2,4)
data[data<0]=0
arr[2,0] 与 arr[2][0] 等价
np.where(conditions,x,y) # 条件判断
np.where(conditions) # 返回输入数组中满足给定条件的元素的索引
```
### 函数运算
```
import numpy.linalg
diag() #返回方阵的对角线元素，或将一维数组转换为方阵
dot() #秩为1的数组，执行对应位置相乘，然后再相加,秩不为1的二维数组，执行矩阵乘法运算
星号（*）乘法运算 # 对数组执行对应位置相乘,对矩阵执行矩阵乘法运算
norm() #求范数，默认为L2范数
multiply() #对应元素位置相乘
trace() #计算对角线元素和
det() #计算矩阵行列式
eig() #计算特征值和特征向量
inv() #计算方阵的逆
qr() #计算QR分解
svd() #计算奇异值分解
solve() #解线性方程组Ax=b
lstsq() #计算Ax=b的最小二乘解

np.dot() # 矩阵乘法，计算矩阵内积
```
# Pandas模块

### Series

```
import pandas as pd
delq = pd.Series([1,0,2,3,0])
delq = pd.Series([1,0,2,3,0],index=['a','b','c','d'])
delq.reindex(['b','a','d','c'],fill_value=0)
delq =pd.Series({'a':1,'b':0,'c':2,'d':3,'e':0})
print delq.values
print delq.index
print delq.dtypes
print delq.get_dtype_counts()
print delq.index.tolist()
print delq[2]
print delq[delq!=0]
```

### DataFrame

```
一个表格型的数据结构。它提供有序的列和不同类型的列值,运算时，会自动对齐行和列，没有重叠引入NaN值
DataFrame的列获取为Series
spend = pd.read_csv('spend.csv',header = 0)
spend.head()
spend.info()
spend.describe()
spend.select_dtypes(include,exclude)
spend._get_numeric_data() #drop non-numeric cols
spend = pd.DataFrame({'a':list(range(10)),'b':list(range(20,10,-1))})
pd.DataFrame.from_dict(data[,orient='index']) 字典转化为dataframe
spend=pd.DataFrame(np.random.randn(4,3),columns=['one','two','three'],index=['a','b','c','d'])
spend['one']['a']=spend.loc['a','one']=spend.loc['a']['one']
spend.loc[:,'one']=spend['one']
spend.loc['a']=spend.loc['a',:]
spend[(spend>n).all(1)] 全部符合条件的行
spend[(spend>n).all(0)] 全部符合条件的列
```

### 索引
```
spend.iloc[3]
#选取第3条记录
spend.ix[3]
#ix可以通过行号和行标签进行索引，而iloc只能通过行号索引,loc只通过行标签索引
spend[(spend.a>5)&(spend.b<=15)]



spend.index.is_unique
# 判断索引是否有重复值

spend.replace(1,'one')
#用‘one’代替所有等于1的值
spend.replace([1,3],['one','three'])
#用'one'代替1，用'three'代替3


spend.rename(columns={'old_name': 'new_ name'})
#选择性更改列名
spend.rename(index=lambda x: x + 1)
#批量重命名索引


spend.set_index('column_one')
#将某列设为索引
spend.reset_index(drop=True,name=)
#重新设定索引列，删除原索引列
spend.reindex(index=[],columns=[],method='ffill')
spend.reindex(spend.index.difference([]))
# 重构索引


spend.drop(index=[])
spend.drop_duplicates(subset=None, keep='first', inplace=False)
#去除特定列下面的重复行

spend.drop(column_name,axis=1)
#Use axis=1（跨列） to apply a method across each row, or to the column labels.
#Use axis=0（跨行）to apply a method down each column, or to the row labels (the index).

```



### 函数应用


```
spend.sum(axis=1,skipna=False)
spend.mode() 众数
spend[col].unique() 
spend.nunique() 获取去重值
spend.idxmax() 获取最大值的索引值
spend.cumsum() 每一列的累加和
spend.insert(loc,col,value) 插入列
spend[col].value_counts() 返回各值频率
spend.apply(pd.Series.value_counts) 查看DataFrame对象中每一列的唯一值和计数

cut将根据值本身来选择箱子均匀间隔，qcut是根据这些值的频率来选择箱子的均匀间隔
pd.qcut(spend,n,labels=[])
pd.cut(spend,n,labels=[])

transform同一时间在一个Series上进行一次转换，返回与 group相同的单个维度的序列
spend.transform({col1: func, col2: func})

col前n个最大小值
spend.nlargest(n, col)
spend.nsmallest(n, col)

apply函数：让函数作用在dataframe某一维的向量
spend[‘a’].apply(lambda x: x+1)
spend.iloc[3].apply(lambda x: x+1)



applymap函数：让函数作用在dataframe的每一个元素上
spend.applymap(lambda x: x+1)


spend.groupby([col1,col2])：返回一个按多列进行分组的Groupby对象
spend.groupby(col1，group_keys=False)[col2].agg(['mean','sum'])
# 返回按列col1分组的所有列的均值,和.group_keys 禁止分组的键

spend.groupby(col1).apply(np.mean)
# apply应用各列，agg仅作用于指定的列,agg可以传入多个函数


spend.pivot_table(index=col1, values=[col2,col3], aggfunc=max)
#创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表

pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)


from patsy import dmatrices
#构建线性模型矩阵
y, X = patsy.dmatrices('y ~ x0 + x1', data，return_type='dataframe')


spend.corr()
spend.corrwith(data.X)
# 变量间的相关系数，和某个确定变量的相关系数

spend.melt(id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)
# 透视操作，id_vars是指普通列的列名，value_vars是指那些需要转换的列名，转化为variable和value列

spend.stack(self, level=-1, dropna=True)
#行头变为列头,参数level指向行索引值，或行索引名称

spend.unstack(self, level=-1, fill_value=None)
#unstack，可使得列头变为行头，参数level指向列索引值，或列索引名称
```


### 排序和合并

```

spend.sort_index([axis=1],[ascending=False],[by=column_name])

spend.sort_values(by = column_name,[ascending = True],[na_position='first'])

pd.merge(df1,df2,how=[inner,outer,left,right],left_on='',right_on='',on=[key1,key2],left_index=false,right_index=false)

pd.concat([s1,s2,s3],axis=，ignore_index=True)
ignore_index 产生新的索引




```
### 缺失值处理

```
pandas默认使用NaN表示缺失数据；
dropna默认丢弃任何含有缺失值的行；
dropna(how='all')只丢弃全为NA的行
fillna(,[inplace=True])替换缺失值,inplace为true直接修改原对象
fillna({1:0.5,3:-1})
isnull() 返回一个布尔值对象，该对象类型与源类型一样
```


# Scikit-learn 模块
Scikit-learn的基本功能主要被分为六大部分：分类，回归，聚类，数据降维，模型选择和数据预处理

### datasets
调用自带数据库中的数据
```
1. datasets.load_*()
2. datasets.fetch_*()
3. datasets.make_*()
```
datasets获取对象的属性：
```
data:数据集
target：数据对应的类标记
target_name:类标记对应的名字
DESCR:数据集的描述信息
```

### preprocessing

```
preprocessing.scale()

preprocessing.StandardScaler().fit_transform()
将数据都聚集在均值0附近，方差值为1。

preprocessing.MinMaxScaler().fit_transform()
将数据矩阵缩放到``[0, 1]``，对于方差非常小的属性可以增强其稳定性；可以维持稀疏矩阵中为0的条目

preprocessing.MaxAbsScaler.fit_transform()
将数据矩阵缩放到``[-1, 1]``

preprocessing.normalize((X, norm='l2'))
归一化的过程是将每个样本缩放到单位范数，使用L1或L2范数，主要思想是对每个样本计算范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的范数等于1

preprocessing.Binarizer().fit_transform(X)
特征二值化，默认阈值为0，大于阈值的分类1，小于阈值的分类0

preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit()
缺失值插补

preprocessing.RobustScaler()
根据第1个四分位数和第3个四分位数之间的范围来缩放数据

```
##### 二值化编码
```
OneHotEncoder(n_values=None, categorical_features=None, categories=None, sparse=True, dtype=<class ‘numpy.float64’>, handle_unknown=’error’)
无法对文本编码，且输入必须是二维
sparse：若为True时，返回稀疏矩阵，否则返回数组
categorical_features：若为all，代表所有的特征都被视为分类特征


LabelEncoder(), 
对不连续的数值或文本进行编码，且输入必须为一维


```
### feature_selection
```
VarianceThreshold(threshold)
通过特征的方差来提取特征，默认方差为0的特征会自动删除

SelectKBest(score_func, k=10)
from minepy import MINE
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
scores按升序排序，选择排前k名所对应的特征,
score_func 通常结合 卡方检验chi2，Pearson相关系数 pearsonr,最大信息系数 MINE

```


### model_selection
主要提供交叉验证和结果评估的工具

```
train_test_split(*array,test_size=0.25,train_size=None,random_state=None,shuffle=True,stratify=None)
# 返回切分的数据集,默认test_size将被设置为0.25，train_size将被设置为0.75，stratify按比例抽取训练集和测试集，random_state不同值获取到不同的数据集


cross_val_score(estimator,raw_data,raw_target,cv,scoring)
# 返回train/test数据集上的每折得分,estimator为分类器，raw_data为原数据，raw_target为原数据类别，cv默认为3折验证,scoring为评分算法

GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, refit=True, cv=’warn’)
#自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，可能会调到局部最优而不是全局最优
param_grid：值为字典或者列表，即需要最优化的参数的取值
score：评价标准，默认None,如scoring='roc_auc'
refit搜索参数结束后，用最佳参数结果再次fit一遍全部数据集
grid.fit()：运行网格搜索
grid_scores_：给出不同参数情况下的评价结果
best_params_：描述了已取得最佳结果的参数的组合
best_score_：成员提供优化过程期间观察到的最好的评分

```

##### k折交叉划分

```
kf=KFold(n_splits=3, shuffle=False, random_state=None)
for train_index,test_index in kf.split():
# 将数据集m划分n_splits个不相交子集，每个子集有m/n_splits个训练样例，每次使用k-1个子集作为训练数据，用1个子集作为测试数据，训练k次。最终的结果是这k次测试结果的均值。返回数据集index

StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
LabelKFold与StratifiedKFold用法相反
```

##### 随机划分法

```
ShuffleSplit()
# 首先对样本全体随机打乱，然后再划分出train/tes对，比KFold交叉更好的控制train/test比例

ss=StratifiedShuffleSplit(n_splits=10,test_size=None,train_size=None, random_state=None)
for train_index, test_index in ss.split(X, y):
# n_splits是将训练数据分成train/test对的组数，StratifiedShuffleSplit是ShuffleSplit的一个变体，返回分层划分


```


### metrics
用于评估方法中，衡量 分类器性能

```
accuracy_score(y_true,y_pred) 计算 accuracy

二分类指标：
precision_recall_curve（y_true,y_score）
fpr,tpr,thresholds=roc_curve(y_true, y_score)
y_score 表示每个测试样本属于正样本的概率，从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本，每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点

roc_auc_score(y_true, y_score, average='macro', sample_weight=None) 计算预测得分曲线下的面积


confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) 输出为混淆矩阵

```


### linear_model
```
LogisticRegression(penalty,dual,C,solver,tol,max_iter)
penalty为正则化范数，默认为L2范数；dual为对偶或者原始方法，通常样本数大于特征数的情况下，默认为False；C为正则化系数λ的倒数，默认为1，值越小，代表正则化越强；solver参数决定了我们对逻辑回归损失函数的优化方法;tol为迭代终止判据的误差范围；max_iter为算法收敛的最大迭代次数

LogisticRegression().fit(X, y, sample_weight=None)
LogisticRegression().predict(X)
LogisticRegression().score(X，Y) 返回准确率
LogisticRegression().predict_proba(X) 返回每个样本每种类别的概率

```

### tree

```
DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,min_samples_leaf =1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,class_weight=None, presort=False)
CART分类回归树算法
tree= DecisionTreeClassifier() 建立决策树模型
tree.fit(X,Y) 构建实例
tree.predict() 预测分类
tree.predict_proba(X) 返回每个样本每种类别的概率


```

### cluster

```
estimator=KMeans(n_clusters,n_init=10,max_iter)
n_cluster为簇的个数，n_init为获取初始簇质心迭代的次数，max_iter为最大迭代次数

estimator.fit() 构建实例
estimator.labels 获取聚类标签
estimator.cluster_centers 获取聚类中心


```

### ensemble

```
RandomForestClassifier(n_estimators,)
n_estimators 决策树的个数，控制模型复杂度，bosstrap 是否有放回，oob_score 没有被boostrap选取的数据是否做验证，max_features 控制选取多少特征量，常用值为\sqrt{n}和log2(n) ,max_depth 控制子树的深度，min_samples_split控制内部节点再划分所需最小样本数，若节点样本数小于值，则不进行划分，min_samples_leaf 控制控制叶子节点最少样本数，若小于值，则和兄弟节点一起被剪枝



IsolationForest(n_estimators=100, max_samples=’auto’, contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=1, random_state=None, verbose=0)
max_samples默认采样数据256条样本
contamination设置样本中异常点的比例
fit(X)
pretict(X) 返回1表示非异常值，-1为异常值
decision_function(X) 返回样本的异常评分，值越小表示越有可能是异常样本


VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('svc', clf3)], voting=['hard','soft'],weight=)
针对分类问题的一种结合策略。基本思想是选择所有机器学习算法当中输出最多的那个类
hard vote 硬投票是选择算法输出最多的标签，软投票是使用各个算法输出的类概率来进行类的选择，输入权重的话，会得到每个类的类概率的加权平均值

```

### neighbors

```
KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)
KNN近邻分类，algorithm有三种，brute’对应第一种蛮力实现，‘kd_tree’对应第二种KD树实现，‘ball_tree’对应第三种的球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法

```

### SVM
```
LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
LinearSVC是线性分类，不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用

NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

OneClassSVM(kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None)
pretict(X) 返回1表示非异常值，-1为异常值
```

### decomposition 
```
PCA(n_components=None, copy=True, whiten=False)
n_components:所要保留的主成分个数n
```
### manifold 
```
TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
n_components:嵌入空间的维度
```
### pipeline 
管道机制实现了对全部步骤的流式化封装和管理

```
Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])

make_pipeline(StandardScaler()，PCA(n_components=2)，LogisticRegression(random_state=1))
pipline的简写

```

# imblearn模块
不平衡数据处理包

### Over_sampling
```
RandomOverSampler(sampling_strategy='auto', return_indices=False, random_state=None, ratio=None)
ros=RandomOverSampler()
X_re,y_re=ros.fit_sample(X, y)

SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, m_neighbors='deprecated', out_step='deprecated', kind='deprecated', svm_estimator='deprecated', n_jobs=1, ratio=None)
kind调参{'regular', 'borderline1', 'borderline2', 'svm'}调参判别k近邻的样本是否属于同类样本
```
### Under_sampling
```
RandomUnderSampler(sampling_strategy='auto', return_indices=False, random_state=None, replacement=False, ratio=None)

NearMiss(sampling_strategy='auto', return_indices=False, random_state=None, version=1, n_neighbors=3, n_neighbors_ver3=3, n_jobs=1, ratio=None)
version调参{1，2，3}调参判别K近邻样本选取

```

### ensemble
```
EasyEnsembleClassifier(n_estimators=10, base_estimator=None, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=1, random_state=None, verbose=0)
从多数类中抽样出和少数类数目差不多的样本，然后和少数类样本组合作为训练集。在这个训练集上学习一个adaboost分类器

```
#  xgboost 模块

### sklearn.XGBClassifier
```
from xgboost.sklearn import XGBClassifier

XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=27,reg_alpha=0, reg_lambda=1, colsample_bylevel=1)

基本参数调优（learning_rate,n_estimators）
决策树特定参数调优(max_depth, min_child_weight, gamma, subsample, colsample_bytree)
正则化参数的调优(lambda, alpha)

参数：booster：指定了用哪一种基模型。可以为：'gbtree','gblinear','dart'
	gamma：最小划分损失min_split_loss。即对于一个叶子节点，当对它采取划分之后，损失函数的降低值的阈值。
	max_depth:树的深度，默认值为6
	min_child_weight： 一个整数，子节点的权重阈值。对于树模型（booster=gbtree,dart），权重就是：叶子节点包含样本的所有二阶偏导数之和。
	subsample：一个浮点数，对训练样本的采样比例。默认值为 1 。如果为0.5，表示随机使用一半的训练样本来训练子树
	colsample_bytree： 一个浮点数，构建子树时，对特征的采样比例。默认值为 1。
	colsample_bylevel： 一个浮点数，寻找划分点时，对特征的采样比例。 默认值为 1。
	reg_alpha： L1 正则化系数
	reg_lambda： L2 正则化系数
	scale_pos_weight： 用于调整正负样本的权重，常用于类别不平衡的分类问题。默认为 1。
	
>> objective 
回归任务
reg:linear (默认)
reg:logistic 
二分类
binary:logistic     概率 
binary：logitraw   类别
多分类
multi：softmax  num_class=n   返回类别
multi：softprob   num_class=n  返回概率
	
	

fit(X, y, sample_weight=None, eval_set=None, eval_metric=None,
    early_stopping_rounds=None,verbose=True, xgb_model=None)

参数：sample_weight： 一个序列，给出了每个样本的权重
	eval_set： 一个列表，元素为(X,y)，给出了验证集及其标签
	xgb_model：一个Booster实例，它给出了待训练的模型。
	eval_metric： 一个字符串或者可调用对象，用于evaluation metric
	early_stopping_rounds：在验证集上，当连续n次迭代，分数没有提高后，提前终止训练

>> eval_metric
回归任务(默认rmse) rmse--均方根误差;mae--平均绝对误差
分类任务(默认error)
auc--roc曲线下面积
error--错误率（二分类）
merror--错误率（多分类）
logloss--负对数似然函数（二分类）
mlogloss--负对数似然函数（多分类）
	
	
predict(data, output_margin=False, ntree_limit=0)
参数：output_margin： 表示是否输出原始的、未经过转换的margin value

predict_proba(data, output_margin=False, ntree_limit=0) ： 执行预测，预测的是各类别的概率

evals_result()： 返回一个字典，给出了各个验证集在各个验证参数上的历史值
```

### Xgboost
```
import xgboost 
xgboost.DMatrix(data, label=None, missing=None, weight=None, silent=False, feature_names=None, feature_types=None, nthread=None) 
无法识别object类型,需使用了sklearn.preprocessing中的LabelEncoder转化
参数：label：一个序列，表示样本标记。，missing： 一个值，它是缺失值的默认值。，weight：一个序列，给出了数据集中每个样本的权重
属性：feature_names： 返回每个特征的名字；feature_types： 返回每个特征的数据类型
方法：.num_col()；.num_row()；.get_label();.get_weight()

xgboost.train()： 使用给定的参数来训练一个booster
xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
   maximize=False, early_stopping_rounds=None, evals_result=None, verbose_eval=True,
   xgb_model=None, callbacks=None, learning_rates=None)
参数：dtrain：DMatrix对象，params： 键值对，num_boost_round： 表示boosting 迭代数量
	evals： (DMatrix,string)验证集，以及验证集的名字，obj：表示自定义的目标函数
	feval： 表示自定义的evaluation 函数，maximize： 如果为True，则表示是对feval 求最大值
	learning_rates： 一个列表，给出了每个迭代步的学习率，
	evals_result： 一个字典，它给出了对测试集要进行评估的指标
	
xgboost.cv()： 使用给定的参数执行交叉验证 
xgboost.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, folds=None,
     metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None,
     fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0,
     callbacks=None, shuffle=True)
     参数：nfold： 表示交叉验证的fold 的数量，stratified： 如果为True，则执行分层采样
     	folds： 一个scikit-learn 的 KFold 实例或者StratifiedKFold 实例
     	shuffle： 如果为True，则创建folds 之前先混洗数据
     	as_pandas： 如果为True，则返回DataFrame；否则返回ndarray

```
