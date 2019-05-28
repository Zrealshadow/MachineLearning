

[TOC]















# Linear Regression

> 参考：
>
> - <https://blog.yaodataking.com/2017/02/28/machine-learning-2/>
> - <https://zhuanlan.zhihu.com/p/30535220>
> - <https://zhuanlan.zhihu.com/p/28408516>



## 理论总结：



### Linear Regression

线性回归用最适直线(回归线)去建立因变量Y和一个或多个自变量X之间的关系。可以用公式来表示：			
$$
Y=w^T*X+b
$$
如何确定w和b？显然，关键在于如何衡量f(x)于y之间的差别。均方误差是回归任务中最常用好的性能度量，因此我们可以试图让均方误差最小化，（之后很多类型的回归都是在损失函数上进行修改）
$$
argmin \sum_{i=1}^m (f(x_i)-y_i)^2
$$

$$
argmin \sum_{i=1}^m (f(x_i)-w*x_i-b)^2
$$

之后根据最小二乘法得到最优解的闭式解，或者利用梯度下降法无限逼近最优闭式解。

（训练过程跳过）



### Ridge Regression

岭回归是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计发，通过放弃最小二乘法的无偏性，以损失部分信息，降低精度为代价获得得回归系数更加符合实际，更可靠的回归方法，**对病态数据的拟合要强于最小二乘法**

损失函数公式如下：
$$
argmin (\sum_{i=1}^m (f(x_i)-y_i)^2+alpha*\sum_{i=1}^m w_i^2）
$$
损失函数中添加的一个惩罚项 $alpha*\sum_{i=1}^m w_i^2$

称为L2正则化。加入此惩罚项进行优化后，限制了回归系数 $w_i$的绝对值，数学上可以证明等价形式如下：
$$
Loss(w)=\sum_{i=1}^m (f(x_i)-y_i)^2
$$

$$
s.t. \sum_{i=1}^m w_i^2<=t
$$

其中t为某个阈值



**岭回归的特点**

当岭参数 alpha=0时，得到的解时最小二乘解

当岭参数alpha 趋向更大时，岭回归系数w趋向于0，约束项t很小

**岭回归限定了所有回归系数的平方和不大于t，在使用普通最小二乘法回归的时候<u>两个变量具有相关性</u>，可能会使得一个系数时很大的正数，另一个系数是一个很大的负数。通过岭回归 的约束条件的现实，可以避免这个问题**



**岭回归的几何意义**

以两个变量为例，残差平方和可以表示 w1,w2的一个二次函数，是一个在三维空间中的抛物面,用等值线来表示。而限制条件$w_1^2+w_2^2<t$，相当于二维平面的一个圆。这个时候等值线与圆相切的点便是在约束条件下的最优点，如下同所示

![image-20190331114712921](../../Library/Application%20Support/typora-user-images/image-20190331114712921.png)





### Lasso Regression

LASSO(The Least Absolute Shrinkage and Selection Operator)是另一种缩减方法，将回归系数收缩在一定的区域内。LASSO的主要思想是构造一个一阶惩罚函数获得一个精炼的模型, **通过最终确定一些变量的系数为0进行特征筛选。**

LASSO的惩罚项为:
$$
s.t. \sum_{i=1}^m |w_i|<=t
$$
与岭回归的不同在于，此约束条件使用了绝对值的一阶惩罚函数代替了平方和的二阶函数。虽然只是形式稍有不同，但是得到的结果却又很大差别。在LASSO中，当alpha很小的时候，一些系数会随着变为0而岭回归却很难使得某个系数**恰好**缩减为0. 我们可以通过几何解释看到LASSO与岭回归之间的不同。

**LASSO回归的几何解释特点**

![image-20190408075654369](../Machine%20Learning/LAB1%20LinearRegression/assets/image-20190408075654369.png)

**筛选变量**

因为约束是一个正方形，所以除非相切，正方形与同心椭圆的接触点往往在正方形顶点上。而顶点又落在坐标轴上，这就意味着符合约束的自变量系数有一个值是 0。

相比圆，方形的顶点更容易与抛物面相交，顶点就意味着对应的很多系数为0，而岭回归中的圆上的任意一点都很容易与抛物面相交很难得到正好等于0的系数。这也就意味着，**lasso起到了很好的筛选变量的作用。**

**复杂度调整**

正放型的大小决定复杂度调整的程度，及惩罚系数t决定复杂度调整的程度，假设t趋近于0，那么这个正方形的大小趋近于一个常数,而所有自变量w的大小趋近于0，这是模型极简情况下的极端情况。及t越小，对参数较多的模型的惩罚程度越大，越容易得到一个简单模型。



### Elastic Net  Regression

其实Lasso回归和Ridge回归都是属于弹性网回归家族，首先先看Elastic Net 回归的损失函数：
$$
argmin (\sum_{i=1}^m (f(x_i)-y_i)^2+0.5*alpha*（1-ratio）*\sum_{i=1}^m w_i^2+alpha*ratio*|w|）
$$
可以看到弹性网回归是将Lasso回归和Ridge回归综合起来，在前边乘了系数，使其对于不同的实际情况有了更广泛的应用性



### Logistc Regression

逻辑回归在是一种解决2分类问题的机器学习方法（0/1问题，但不限于此问题softmax回归是在logistc回归的基础上解决多分类问题），在实际案例中运用较广。逻辑回归是一种广义的线性模型，去除sigmoid函数映射，就是一个线性回归模型。逻辑回归通过Sigmoid函数引入了非线性因素，因此可以轻松处理0/1分类问题。
$$
Y=sigmoid(w^T*X+b)
$$
**sigmoid 函数**
$$
g(z)=\frac{1}{1+e^{-z}}
$$
如图：

![image-20190503140025785](../Machine%20Learning/LAB1%20LinearRegression/assets/image-20190503140025785.png)

取值在0，1之间，这个特性对于解决2分类问题极为重要。这个函数在给定的w，b的条件下被认为是X的取1的概率，即：
$$
P(Y=1|w,b)=\frac{1}{1+e^{-(w^T*X+b)}}
$$
如果该值大于0.5，及划分为1，否则为0

（注：选择0.5为阈值是一个一般的做法，实际应用时特定的情况可以选择不同阈值，如果对正例的判别准确性要求高，可以选择阈值大一些，对正例的召回要求高，则可以选择阈值小一些。）



> 在softmax中，如果划分n类，所得结果为一个1维向量，其中概率最大的为该数据项所划分的类别。



## 应用实例

> 因为实验过程简单，下面重点介绍API的相关参数作用，注意事项，以及结果分析
>
> 实验源码：[https://github.com/Zrealshadow/MachineLearning/tree/master/LAB1%20LinearRegression](https://github.com/Zrealshadow/MachineLearning/tree/master/LAB1 LinearRegression)
>
> 实验步骤解释在注释中有明显说明

### california_housing regression

> 分别使用Linear Regression ， Ridge Regression， Lasso Regression，ElasticNet Regression 对房价进行预测

读取数据：

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
data=fetch_california_housing()
data_x=data.data
data_y=data.target
feature=data.feature_names
x_train,x_test,y_train,y_test=train_test_split(\
                        data_x,data_y,test_size=0.2,random_state=1)
```

其中fetch_california_housing()返回一个class，成员主要成员data，target，feature_names，分别代表，自变量，因变量和特征描述

train_test_split 是一个常用的划分训练测试集的常用函数，其中主要参数如下：

> Arrays: x,y	输入的数据集
>
> test_size: float 0.0-1.0之间，表示测试集占总训练样本的比例 default:0.25
>
> train_size: float 0.0-1.0之间，表示训练集占总训练样本的比例
>
> random_state:  int 随机种子， default：None
>
> shuffle: bool  分裂之前是否打乱数据集



**Linear Regression**

该类在：

```python
from sklearn.linear_model import LinearRegression
```

定义时主要参数：

> fit_intercept: bool 	是否对加载入模型的数据进行处理
>
> normalize：bool 如果fit_intercept 为False该参数忽略，否则若normalize 为ture， 对输入数据进行L2正则化
>
> copy_X：bool，运算过程中是否重写x

方法：

> fit(), 	fit(X,y) 对模型进行训练
>
> get_param()	返回模型参数
>
> predict()	predict(test_x)利用模型进行预测，返回Y
>
> score(X,y)	利用模型均行预测，并对模型正确率进行打分，打分公式为R2_score



*在后面API中不会重复上述相同功能的参数或方法，只对特别API指出*



**Ridge Regression / Lasso Regression**

```python
from sklearn.linear_model import Ridge,Lasso
```

参数：

> 同Linear Regression
>
> max_iter:	最大迭代次数
>
> alpha：	惩罚系数，为正则项前的系数



**ElasticNet regression**

```python
from sklearn.linear_model import ElasticNet
```

参数：

> 同Linear Regression
>
> max_iter:	最大迭代次数
>
> alpha：	惩罚系数，为正则项前的系数
>
> L1_ratio:float  调整L1正则项和L2正则项之间比例



网格参数调优 GridSearch

```python
from sklearn.model_selection import GridSearchCV
```

主要参数：

> estimator:评估标准，及定义一个损失函数，模型
>
> param_grid： dict 候选参数
>
> cv：int 几则交叉验证 

返回对象成员：

> **cv_results_** :data frame / np.array/list 返回所有参数组合后的结果
>
> best_score_:float 交叉运算中的最佳得分
>
> best_param_:dict	最好的参数组合



### 20newsgroups classify

> 利用logistic对数据进行分类

获得数据集：

```python
from sklearn.datasets import fetch_20newsgroups_vectorized
    
# =============================================================================
#  导入新闻数据
# =============================================================================
data_logistic=fetch_20newsgroups_vectorized()
data_log_x=data_logistic.data
data_log_y=data_logistic.target

# =============================================================================
# 分割训练数据集
# =============================================================================
x_train,x_test,y_train,y_test=train_test_split(data_log_x,data_log_y,test_size=0.2,random_state=1)
```

**logistic regression**

```python
from sklearn.linear_model import LogisticRegression

```

参数：

> **penalty：**惩罚项，str类型，可选参数为l1和l2，默认为l2。用于指定惩罚项中使用的规范。newton-cg、sag和lbfgs求解算法只支持L2规范。L1G规范假设的是模型的参数满足拉普拉斯分布，L2假设的模型参数满足高斯分布，所谓的范式就是加上对参数的约束，使得模型更不会过拟合(overfit)，但是如果要说是不是加了约束就会好，这个没有人能回答，只能说，加约束的情况下，理论上应该可以获得泛化能力更强的结果。
> **dual：**对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。
> **tol：**停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。
> **c：**正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
> **fit_intercept：**是否存在截距或偏差，bool类型，默认为True。
> **intercept_scaling：**仅在正则化项为”liblinear”，且fit_intercept设置为True时有用。float类型，默认为1。
> **class_weight**：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考虑权重，即为None。如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。举个例子，比如对于0,1的二元模型，我们可以定义class_weight={0:0.9,1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。如果class_weight选择balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当class_weight为balanced时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))。n_samples为样本数，n_classes为类别数量，np.bincount(y)会输出每个类的样本数，例如y=[1,0,0,1,1],则np.bincount(y)=[2,3]。 
> 那么class_weight有什么作用呢？ 
> 在分类模型中，我们经常会遇到两类问题：
> 第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。这时，我们可以适当提高非法用户的权重。
> 第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。
> **random_state：**随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
> **solver：**优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是： 
> 1.liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
> 2.lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
> 3.newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
> 4.sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
> 5.saga：线性收敛的随机优化算法的的变重。
>
> <u>总结：</u> 
> <u>liblinear适用于小数据集，而sag和saga适用于大数据集因为速度更快。</u>
> <u>对于多分类问题，只有newton-cg,sag,saga和lbfgs能够处理多项损失，而liblinear受限于一对剩余(OvR)。啥意思，就是用liblinear的时候，如果是多分类问题，得先把一种类别作为一个类别，剩余的所有类别作为另外一个类别。一次类推，遍历所有类别，进行分类。</u>
> <u>newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear和saga通吃L1正则化和L2正则化。</u>
> <u>同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。</u>
> <u>从上面的描述，大家可能觉得，既然newton-cg, lbfgs和sag这么多限制，如果不是大样本，我们选择liblinear不就行了嘛！错，因为liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。</u>
>
> 
>
> **max_iter**：算法收敛最大迭代次数，int类型，默认为10。仅在正则化优化算法为newton-cg, sag和lbfgs才有用，算法收敛的最大迭代次数。
> **multi_class**：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。 
> OvR和MvM有什么不同*？* 
> OvR的思想很简单，无论你是多少元逻辑回归，我们都可以看做二元逻辑回归。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类样本以外的所有样本都作为负例，然后在上面做二元逻辑回归，得到第K类的分类模型。其他类的分类模型获得以此类推。
> 而MvM则相对复杂，这里举MvM的特例one-vs-one(OvO)作讲解。如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。
> 可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而MvM分类相对精确，但是分类速度没有OvR快。如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。
>
> **verbose**：日志冗长度，int类型。默认为0。就是不输出训练过程，1的时候偶尔输出结果，大于1，对于每个子模型都输出。
> **warm_start**：热启动参数，bool类型。默认为False。如果为True，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）。
>
> **n_jobs**：并行数。int类型，默认为1。1的时候，用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。为-1的时候，用所有CPU的内核运行程序。
>
> 参考：<https://blog.csdn.net/jark_/article/details/78342644>



# DecisionTree And RandomForest



## DecisionTree



### 划分标准

> 我们都知道在ID3，和C4.5决策树中，都是基于信息论来对选取最优特征的，那么这个信息熵的计算公式为什么是这样的？下面介绍信息熵公式的产生



**---------------------------------------------------------信息量---------------------------------------------------------**



信息量是对信息的度量，就跟时间的度量是秒一样，当我们考虑一个离散的随机变量x的时候，当我们观察到的这个变量的一个具体值的时候，我们接收到了多少信息呢？

多少信息用信息量来衡量，**我们接受到的信息量跟具体发生的事件有关。**信息的大小跟随机事件的概率有关。

**越小概率的事情发生了产生的信息量越大**：湖南产生的地震了

**越大概率的事情发生了产生的信息量越小**，如太阳从东边升起来了

（是不是很玄学，但是确实很有道理）

**结论:一个具体事件的信息量应该是随着其发生概率而递减的，且不能为负。**

那么问题又来了，这种减函数这么多为什么要用下面的log公式来度量？
$$
info(D)=-\sum{p_i}*\log_2(p_i)
$$
**如果我们有俩个不相关的事件x和y，那么我们观察到的俩个事件同时发生时获得的信息应该等于观察到的事件各自发生时获得的信息之和，即：**
$$
info(x,y)=info(x)+info(y)
$$
由于x，y是俩个不相关的事件，那么满足 $p(x,y)=p(x)*p(y)$

根据上面推导，**我们很容易看出info(x)一定与p(x)的对数有关（因为只有对数形式的真数相乘之后，能够对应对数的相加形式，可以试试）**。因此我们有信息量公式如下：
$$
info(x)=-\log_2{p(x)}
$$
**下面解决俩个疑问？**

（1）为什么有一个负号

**其中，负号是为了确保信息一定是正数或者是0，总不能为负数吧！**

（2）为什么底数为2

**这是因为，我们只需要信息量满足低概率事件x对应于高的信息量。那么对数的选择是任意的。我们只是遵循信息论的普遍传统，使用2作为对数的底！**



**---------------------------------------------------------信息熵---------------------------------------------------------**



**信息量度量的是一个具体事件发生了所带来的信息，而熵则是在结果出来之前对可能产生的信息量的期望——考虑该随机变量的所有可能取值，即所有可能发生事件所带来的信息量的期望。即**
$$
H(x)=-\sum(p(x)*info(x))
$$
即：
$$
Info(x)=-\sum{p(x)*log_2p(x)}
$$


**信息增益度**

信息增益度是ID3决策树的选择最优划分特征的判断方法，它的思路是：
$$
Gain(D,A)=Entorpy(D)-\sum_{v=1}^{V}\frac{|D^v|}{|D|}*Entropy(D^v)
$$
**选择A特征之前呈现的信息熵Entropy  与 选择A特征之后呈现的信息熵Entorpy(A) 之差 Gain(A),若Gain(A) 越大，说明选择A特征后得到的信息越多，因此选择A特征为最优划分特征**



**---------------------------------------------------------信息增益率---------------------------------------------------------**

信息增益度是C4.5决策树的选择最优划分特征的判断方法，它的思路：
$$
IV(a)=-\sum_v^{V}\frac{|D^v|}{|D|}log{\frac{|D^v|}{|D|}}								
$$

$$
Gain\_ratio=\frac{Gain(D,A)}{IV(A)}
$$

为什么要引入信息增益率的概念，是因为在ID3决策树中，当一个特征他又很多个划分，每个划分下样本的量很少，这时整个样本纯度很高，选择该特征得到的信息增益度会很高，因此在ID3的判断标准下有选择划分很多的特征的趋势。但是较多取值属性带来的问题是，整个模型的泛化能力很弱。因此引入了IV来减少因为划分数量而带来的信息增益度的降低。

**实际操作中，通常选取信息增益度TOPN的特征，然后分别计算它们的信息增益率，选择信息增益率最大的为最优划分特征**



**---------------------------------------------------------基尼指数---------------------------------------------------------**



基尼指数是CART树的的选择最优划分特征的判断方法，思路如下：

基尼指数是为了信息熵模型中的对数运算，而且保留熵模型的特点(即代表该事物的信息量)而产生的模型。

**基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。这和信息增益(比)是相反的。**
$$
Gini(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^{k}p_k^2
$$

$$
Gini(D,A)=\sum_{v=1}^{V}\frac{|D^v|}{|D|}Gini(D_v)
$$

CART树在使用Gini指数时，还额外进行了简化，在CART树下，所有特征只能进行2分类，即使例如特征A有三个划分（A1，A2，A3);在Cart树中，也会将把A分成{A1}和{A2,A3}{A1}和{A2,A3}, {A2}和{A1,A3}{A2}和{A1,A3}, {A3}和{A1,A2}{A3}和{A1,A2}三种情况，找到基尼系数最小的组合，比如{A2}和{A1,A3}{A2}和{A1,A3},然后建立二叉树节点，一个节点是A2对应的样本，另一个节点是{A1,A3}对应的节点。同时，由于这次没有把特征A的取值完全分开，后面我们还有机会在子节点继续选择到特征A来划分A1和A3。



### 过拟合与剪枝



**---------------------------------------------------------过拟合---------------------------------------------------------**

从直观上来说，只要决策树足够深，划分标准足够细，它在训练集上的表现就能够接近完美；但同时也容易像想象，由于它可能吧训练集的一些"特性"当作所有数据的"共性"来看待，因此它在位置数据上的表现可能比较一般，亦即会出现过拟合的问题。

模型出现过拟合问题一般是因为模型太过复杂。所以决策树解决过拟合问题就是采取适当的"剪枝"。剪枝主要分为"预剪枝（pre-pruning)"和"后剪枝（post-pruning）"。



**---------------------------------------------------------预剪枝---------------------------------------------------------**

在建立决策树过程中，其实已经进行了预剪枝的过程，就是我们建树的"停止条件"。通常来说，在建树过程中，我们的停止条件是划分到，当前划分下无其他类别或者已经没有划分特征为止。

当然我们也可以采用交叉验证的方法进行预剪枝，当选出一个划分特征后，我们通过验证集的验证得到在该验证集下的正确率，如果该正确率大于划分前验证集的正确率或者给定的阈值正确率则对其进行划分。否则停止划分。



**---------------------------------------------------------后剪枝---------------------------------------------------------**

一般提起的剪枝都是指的后剪枝，它是指在决策树生成完毕后再对其进行修剪，把多余的节点剪掉。换句话说后剪枝是从全局出发，通过某种标准对一些Node进行局部剪枝。从而有效的减少模型的复杂度。通常有两种做法：

- 应用交叉验证的思想，若局部剪枝能够使得模型在测试集上的错误率降低，则进行局部剪枝（预剪枝中也有类似思想，但不同之处是，后剪枝是从低向上，而预剪枝是从顶至下）。

- 应用正则化思想，综合考虑**不确定性**和**模型的复杂程度**来确定一个新的损失，用该损失来作为一个Node是否进行局部剪枝的标准。定义新的损失通常如下
  $$
  C_a(T)=C(T)+a|T|
  $$

  > 其中C(T)即是该Node和不确定性相关的损失，|T|则是该Node下属叶节点的个数,a为惩罚因子。不妨设第t个叶节点含有Nt个样本且这Nt个样本的不确定性为Ht，那么新算是一般可以直接定义为加权不确定性
  > $$
  > C(T)=\sum_{t=1}^{|T|}N_t*H_t(T)
  > $$

  - C4.5 ，ID3 剪枝方法： 通过比较剪枝前和局部剪枝后的C(T)来决定是否剪枝
  - CART 树剪枝方法有点奇怪： 它是取一系列阈值[C1,C2,C3,…..,Cn]，通过C(T)与阈值比较进行局部剪枝，最后每个阈值C会得到一颗决策树 T，得到一些列决策树[T1,T2,T3…..Tn]，对这一些列决策树进行交叉验证，正确率最高的为最终生成的决策树。



**---------------------------------------------------------剪枝比较---------------------------------------------------------**

一般情况下，后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树，但后剪枝过程实在生成完全决策树后进行的，并且要自底向上地对树中的所有非叶节点进行一一考察，因此训练时间开销比未剪枝决策树和预剪枝决策树都要大得多。



### Sklearn中的决策树

> 实验源码：
>
> [https://github.com/Zrealshadow/MachineLearning/tree/master/LAB2%20DecisionTree](https://github.com/Zrealshadow/MachineLearning/tree/master/LAB2 DecisionTree)



**重点：**

sklearn中的决策树都是CART树，虽然在参数中有criterion的算法，但是选择"gini"或者"entropy"只是度量方式不同，其结构都是二叉CART树。

> scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.
>
> ——[官方文档](https://scikit-learn.org/stable/modules/tree.html)

sklearn用的是优化后的CART树，至今不提供对决策树种类选择的接口

**---------------------------------------------------------决策树分类---------------------------------------------------------**



```python
sklearn.tree.DecisionTreeClassifer
```

参考：

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

https://www.cnblogs.com/pinard/p/6056319.html

| 参数                                             | DecisionTreeClassifier                                       |
| ------------------------------------------------ | ------------------------------------------------------------ |
| 特征选择标准criterion                            | 可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。一般说使用默认的基尼系数"gini"就可以了。 |
| 特征划分点选择标准splitter                       | 可以使用"best"或者"random"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random" |
| 划分时考虑的最大特征数max_features               | 可以使用很多种类型的值，                                                                                      默认是"None",意味着划分时考虑所有的特征数；                                                如果是"log2"意味着划分时最多考虑log2Nlog2N个特征；                                  如果是"sqrt"或者"auto"意味着划分时最多考虑N‾‾√N个特征。                           如果是整数，代表考虑的特征绝对数。                                                                  如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。                                                                                           一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。 |
| 决策树最大深max_depth                            | 决策树的最大深度，默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。 |
| 内部节点再划分所需最小样本数min_samples_split    | 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。 |
| 叶子节点最少样本数min_samples_leaf               | 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。之前的10万样本项目使用min_samples_leaf的值为5，仅供参考。 |
| 叶子节点最小的样本权重和min_weight_fraction_leaf | 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。 |
| 最大叶子节点数max_leaf_nodes                     | 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。 |
| 类别权重class_weight                             | 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"None" |
| 节点划分最小不纯度min_impurity_split             | 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。 |
| 数据是否预排序presort                            | 这个值是布尔值，默认是False不排序。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为true可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，我速度本来就不慢。所以这个值一般懒得理它就可以了。 |





**---------------------------------------------------------决策树回归---------------------------------------------------------**

```python
sklearn.tree.DecisionTreeRegressor
```

在回归树中，API 参数于分类树大致相同，不同点有如下两种

| 参数                  | DecisionTreeRegression                                       |
| --------------------- | ------------------------------------------------------------ |
| 特征选择标准criterion | 可以使用"mse"或者"mae"，前者是均方差，后者是和均值之差的绝对值之和。推荐使用默认的"mse"。 |
| 类别权重class_weight  | 不适用于回归树                                               |



**---------------------------------------------------------决策树可视化---------------------------------------------------------**

决策树可视化用到了`graphviz` `pydotplus`两个包

```python
#决策树回归建模并评估
regression_tree=DecisionTreeRegressor(max_depth=3,random_state=1)
regression_tree.fit(x_train,y_train)
print("2 feature selection:\n")
evalue(regression_tree,y_test,x_test)


#决策树可视化
dot_data=export_graphviz(regression_tree,out_file=None,\
                        feature_names=[features[k[0]],features[k[1]]],rounded=True,filled=True)
'''
export_graphviz(decision_tree,out_file,max_depth,feature_names,class_name,label,filled,leaves_paralled,impurity,node_ids)
@param decision_tree: 可视化的决策树模型
@param out_file: 命名输出文件，默认为None
@param max_depth:最大深度，默认为None 所有可视化
@param feature_names: 特征名称list
@param class_name: 每个特征下的分类的名称 list
@param label: 是否在节点显示不纯度 [all，None,root]
@param filled: True显示分类的主要类，特别是回归的端点值，有多输出的节点的纯度
@param leaves_paralled: True将所有叶子节点展示在树的底端
@param impurity: True 显示每个node的不纯度
@param node_ids: True 显示每个node的编号
@return : dot_data :String
'''
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png('./regressiontreeOf2Features.png')
```





**---------------------------------------------------------决策树算法优缺点---------------------------------------------------------**

**优点**

- 模型简单直观，适用于展示
- 基本不需要预处理
- 既可以处理离散值，也可以处理连续值
- 可以处理多分类问题
- 可以交叉验证进行剪枝，提高泛化能力
- 对异常点的容忍性好，健壮性高

**缺点**

- 容易过拟合
- 随着样本的变动而变动
- 难以寻找最优决策树
- 对于复杂的关系，决策树难以学习



------





## RandomForest



### 理论介绍

随机森林是集成学习的一种算法，它是Bagging算法中的一种，下面将先介绍Bagging算法的步骤，再介绍随机森林的过程。集成学习的思想在大数据竞赛中是十分常见的，因为通常集成模型比单个模型的性能更加优越。Bagging算法的主要思路：**将相同类型但参数不同的弱分类器进行提升**



**---------------------------------------------------------Bootstrap---------------------------------------------------------**



Bootstrap是数理统计中非常重要的一个理论，这里将不会详细介绍其原理，只介绍其过程：

Bootstrap做法：

1. 从$X$(一个大的样本集)中随机抽出一个样本(每个样本的几率相同)
2. 将该样本的拷贝放入数据集$X_j$
3. 将该样本放回$X$中
4. 以上三个步骤重复N次，从而使得$X_j$中有N个样本。这个过程将对j=1,2,3,…..M都进行一遍，从而我们最终能得到M个含有N个样本的数据集$X_1,X_2,X_3……X_M$



**---------------------------------------------------------Bagging---------------------------------------------------------**



Bagging 算法的全程是 **Bootstrap Aggregating** ，其思想非常简单

1. 用Bootstrap 生成除M个数据集
2. 用着M个数据集训练出M个弱分类器
3. 最终模型即位这M个弱分类器的简单组合

所谓简单组合：

1. 对于分类问题使用简单的投票表决
2. 对于回归问题则进行简单的取平均

Bagging 算法的特点：

对于"不稳定"（对训练集敏感：若训练样本变化，结果会发生较大变化）的分类器，Bagging算法能够显著的进行提升，这是被大量实验进行证明的。**而且，弱分类器之间的"差异"越大，提升效果更为明显**



**---------------------------------------------------------RandomForest---------------------------------------------------------**

上文中介绍了决策树模型的缺点：很容易受到样本变化带来的影响，因此我们用Bagging算法对其进行提升便成为了随机森林算法。很容易猜到在随机森林中，Bagging中的弱分类器M就是一个CART树。算法流程：

- 用Bootstrap生成M个数据集
- 用这M个数据集训练出M课**不进行后剪枝**的决策树，且在每颗决策树的生成过程中，对Node进行划分时，都从可选特征D个中随机挑选出K个特征，然后依着划分特征的一句从K个特征中选出最优特征作为划分标准
- 最终模型即为这M个弱分类器的简单组合。



> 随机森林除了像Bagging算法那样对样本进行随机采样以外，随机森林还**对特征进行了某种意义上的随机采样**。这样做的意义是通过对特征引入随机扰动，可以使个体模型之间的差异进一步增加，从而提升最终模型的泛化能力。



### Sklearn中的随机森林

> 实验源码：
>
> [https://github.com/Zrealshadow/MachineLearning/tree/master/LAB2%20DecisionTree](



```python
class sklearn.ensemble.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)

'''

---Paramter---

@param: n_estimators: integer (default=100)
森林中CART树的个数，弱分类器个数

@param: criterion: string (default="gini") ["gini","entropy"] 
选择使用gini不纯度来度量还是信	息熵来度量

@param: max_depth: integer or None (default=None) 
选择树的最大深度，否则深度无限制

@param: min_sample_split: int,float,(default=2) 
限制子树继续划分，某节点样本数小于么min_sample_split则停止划分

@param: min_samples_leaf : int, float, optional (default=1) 
叶子节点最少样本数目

@param：min_weight_fraction_leaf : float, optional (default=0.)
叶子节点最小样本权重，小于这个值则被剪枝

@param: max_features : int, float, string or None, optional (default="auto")
				- If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        最大特征数
        
@param: max_leaf_nodes : int or None, optional (default=None)
最多叶子节点数，超出的叶子节点将按照不纯度进行剪枝

@param: min_impurity_decrease : float, optional (default=0.) 
最小减少不纯度阈值，如果划分节点的不纯度的减少量大于等于该值则进行划分

@param: min_impurity_split : float, (default=1e-7) 
最小不纯度阈值，如果一个节点要进行分裂，其不纯度要大于min_impurity_split

-----------------RF框架参数------------------
@param: bootstrap : boolean, optional (default=True) 
是否使用bootstrap，如果否，利用所有数据集进行训练

@param: oob_score : bool (default=False) 
是否采用袋外样本来评估模型的好坏,带外样本可以增加模型泛化能力

@param: n_jobs : int or None, optional (default=None) 
使用几个事件，感觉应该和分布式有关


---attribute---

@attribute: estimators_ : list of DecisionTreeClassifier 
返回所有子树的列表

@attribute:n_classes_ : int or list
分类个数

@attribute:n_features_ : int
特征个数

@attribute:feature_importances_: array of shape = [n_features]
特征重要程度的array，数值越高，特征越重要

@attribute:oob_score_ : float
采用打底外样本评估模型的分数，当param中oob_score=true时存在

'''

class sklearn.ensemble.RandomForestRegressor(n_estimators=’warn’, criterion=’mse’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)

'''
大部分参数与上述相同：
@param: criterion: string, optional (default="mse") ["mse","mae"]
选择标准，方差还是绝对误差
'''

```



# Support Vector Machine



## The Theory

> 报告中大部分理论推导部分参考：书籍：《Python与机器学习实战》——何宇健， 并结合自己的理解进行阐述。



### Perception And Gradient Decent



用数学的描述线性可分

存在一个超平面π能将数据集D中的正负样本精确的划分到π两侧，及：
$$
\exists \Pi : w\cdot x+b=0
$$
使得：
$$
w\cdot x_i+b<0 (\forall y_i=-1)
$$

$$
w\cdot x_i+b>0 (\forall y_i=1)
$$



感知机算法的目的是找到能够将线性可分的数据集中正负样本点精确划分到两侧的超平面π。考虑到机器学习的共性，将寻找超平面的过程转化为最小化一个损失函数的过程：
$$
L(w,b,x,y)=-\sum_{x_i\in E}y_i(w\cdot x_i+b)
$$
其中E为误分类点的集合。

该损失函数及为所有误分类点到超平面的距离

如何优化$L(w,b,x,y)$使其最小，我们使用启发式的方法梯度下降法来逼近最优解，此处不详细介绍随机梯度下降法。

对w和b的求偏导
$$
\frac{\partial L(w,b,x_i,y_i)}{\partial w}=
\begin{equation}
\left\{
\begin{array}{rcl}
0&&&(x_i,y_i)\notin E\\
-y_ix_i&&&(x_i,y_i)\in E
\end{array}
\right.
\end{equation}
$$

$$
\frac{\partial L(w,b,x_i,y_i)}{\partial b}=
\begin{equation}
\left\{
\begin{array}{rcl}
0&&&(x_i,y_i)\notin E\\
-y_i&&&(x_i,y_i)\in E
\end{array}
\right.
\end{equation}
$$

感知机算法流程

> 初始化参数：
> $$
> w=(0，....,0)^T  ,b=0
> $$
>  对j=1,…..M:
> $$
> E=\{(x_i,y_i)|y_i(w\cdot x_i +b)<=0\}
> $$
> E为误分类点的集合，若E=空集 退出循环
>
> 否则，任取E中一个点，利用它进行更新参数
> $$
> w_{new}=w+\eta y_i x_i
> $$
>
> $$
> b_{new}=b+\eta y_i
> $$
>
> 最后输出：感知机模型
> $$
> g(x)=sign(w\cdot x+b)
> $$



**-----------------------------------------------------------感知机算法对偶形式----------------------------------------------------------**

> 对偶形式的表示在SVM的核函数的理解中有很大的作用
>
> 在我看来，这种表现形式比较新颖，是另外一种理解机器学习的角度

感知机算法，SVM算法包括机器学习的很多算法在内，都是为了最小化定义的损失函数。在数学上及是**有约束的最优化问题**。在数学上，为了便于求解此类问题，我们常常会利用**拉格朗日对偶性**来将原始问题转化为更好解决的对偶问题。而且对偶问题也存在一定的共性。

下面介绍感知机算法的对偶形式的转化：

从感知机的参数更新策略中我们可以得知：参数更新是完全基于样本点的。考虑到我们要将参数w和b表示为样本点的线性组合，一个自然的想法就是记录下在核心步骤中各个样本分别被利用了多少次，然后利用这个次数来将w和b表示出来

若设样本点$（x_i,y_i)$一共在核心步骤中被利用了$n_i$次，那么就有当初始化参数w=(0，0，0,….,0)时:(结合梯度下降发的更新策略)
$$
w=\eta \sum_{i=1}^N n_iy_ix_i
$$

$$
b=\eta\sum_{i=1}^N n_iy_i
$$

如果进一步设$\alpha _i=\eta n_i$则有：
$$
w=\sum_{i=1}^N\alpha_i y_i x_i
$$

$$
b=\sum_{i=1}^N \alpha_i y_i
$$

将其带入感知机模型则可得到感知机的对偶算法：

> 初始化参数：
> $$
> \alpha=(\alpha _1，....,\alpha_n)^T=(0,......,0)^T 
> $$
>  对j=1,…..M:
> $$
> E=\{(x_i,y_i)|y_i(\sum_{k=1}^N\alpha_k y_k(x_k \cdot x_i+1))<=0\}
> $$
> E为误分类点的集合，若E=空集 退出循环
>
> 否则，任取E中一个点，利用它进行更新参数
> $$
> \alpha_{new}=\alpha+\eta
> $$
> 最后输出：感知机模型
> $$
> g(x)=sign(\sum_{k=1}^N \alpha_k y_k(x_k \cdot x_i+1))
> $$

需要指出的是在该式子中x仅以内积的形式出现；这个性质将与后面的核技巧联系，能够将 线性算法升级成非线性算法。



> 个人总结一下感知机的对偶形式，其主要思想就是将 算法中更新参数的方式转换成了，记录每个样本点分别被利用了多少次，整个的参数更新也是在更新这个次数的向量(虽然其中更新的是 $\alpha$，但是一般情况下学习速率不变的时候，其更新的本质就是次数，每一次更新次数向量中的一个维度+1 )。最后根据这个次数的向量，然后结合数据集计算出训练完成的感知机模型。
>
> 虽然感知机算法十分的容易，但是它是SVM算法的基础。其实了解SVM可以知道，SVM是一种泛化能力更强的感知机，就线性核SVM而言，其求解思路与感知机相当，都是**有约束的最优化问题**，只是相对于SVM，这个约束相较而言，更多更复杂。





### 线性SVM

svm的出现要提到感知机的缺陷，从上述解法中很容易看出，感知机的解是有无穷多个的，因为感知机只需满足成功分类所有的训练集样本即可，只要训练集中所有的数据分类正确就停止训练，换句话说：**感知机没有考虑到模型的泛化能力**。

对于这个问题SVM提出了改进的思路，一个想法：**在训练过程中考虑超平面到点集的距离，并努力让这个距离最大化**。因此我们定义点到超平面的几何距离：
$$
d(x_i,\Pi)=\frac{1}{||w||}|w \cdot x_i+b|
$$
根据SVM的思路，努力让这个距离最大化，及设最大化距离 d，使得
$$
\frac{1}{||w||}y_i(w \cdot x_i+b)>=d(i=1,....,N)
$$
该问题等价，最大化 $\frac{d^*}{||w||}$，这个$d^*$被定义为函数距离，使得：
$$
y_i(w \cdot x_i+b)>=d^*(i=1,....,N)
$$

> 这一步骤后很多资料上都写的，不妨设d* 为1，但是并没有说清楚 为什么 要取1。
>
>  其实分析可以发现这里的d*的取值对该优化问题的解是没有任何影响的。
> $$
> d^*=d\cdot||w||
> $$
> 当d* 变成k d*时,在超平面不变的情况下：
> $$
> kw\cdot x_i+kb=kd^*
> $$
> 及w和b也会变成相应的kw，kb。此时d 和不等式的约束(只是乘了一个系数)都没有变化。所以对于优化问题没有影响。因此不妨设d*=1。



因此优化问题就是最大化距离$d=\frac{1}{||w||}$,约束条件是：
$$
y_i(w \cdot x_i+b)>=1(i=1,....,N)
$$


该优化问题又可以转化成
$$
\begin{equation}  
             \begin{array}{**lr**}  
             Min(\frac{1}{2}{||w||}^2)  \\  
               y_i(w \cdot x_i+b)-1>0(i=1,....,N)
             \end{array}  
\end{equation}
$$
只要训练集D线性可分，那么SVM算法对于这个优化问题的解就存在唯一性；数学上可矣证明。

假设该优化问题的解为$w^*$和$b^*$,那么我们得到的超平面：
$$
\Pi^*:w^*\cdot x+b^*=0
$$
考虑不等式的约束条件：
$$
\begin{equation}  
             \begin{array}{**lr**}  
                \Pi_1:&w^* \cdot x+b^*=-1  \\  
               \Pi_2:&w^* \cdot x+b^*=1
             \end{array}  
\end{equation}
$$
在上面两个超平面之间是没有样本点的，通常称这两个超平面为**间隔边界**,称在其上的样本点的向量为**支持向量**



考虑到最大间隔的思想，我们上述的所有说明是当其函数间隔d*=1时，这是"硬间隔"，其实可以做一定的拖鞋"妥协"：将"硬"间隔转化为更加普适的"软"间隔。从数学的角度来说，这等价于将不等式约束条件放宽：
$$
\begin{equation}  
             \begin{array}{**lr**}  
            Min(\frac{1}{2}||w||^2)  \\ 
              y_i(w \cdot x_i+b)>1-\zeta_i(i=1,....,N)\\
              \zeta_i>=0(i=1,...,N)
             \end{array}  
\end{equation}
$$
我们希望用随机梯度的方法对其进行求解，需要将问题近似转化为一个无约束的最优化问题：
$$
\zeta=l(w,b,x,y)=max(0,1-y(w\cdot x+b))
$$
然后上式子就可以把完整的损失函数写成
$$
L（w,b,x,y)=\frac{1}{2}{||w||}^2+C\sum_{i=1}^Nl(w,b,x_i,y_i)
$$
其中C为惩罚因子，它反映了对松弛变量$\zeta_i$的惩罚力度，C越大，意味着最终的SVM模型越不能容忍**误分类的点**或**在最大间隔之间的点**。

通过求L(w,b,x,y)最小值，来求解上述SVM的最优化问题。于是我们利用随机梯度下降的算法，对其求偏导：
$$
\frac{\partial L(w,b,x_i,y_i)}{\partial w}=w+
\begin{equation}
\left\{
\begin{array}{rcl}
0&&&y_i（x_i\cdot x_i+b)>=1\\
-Cy_ix_i&&&y_i（x_i\cdot x_i+b)<1
\end{array}
\right.
\end{equation}
$$

$$
\frac{\partial L(w,b,x_i,y_i)}{\partial b}=
\begin{equation}
\left\{
\begin{array}{rcl}
0&&&y_i（x_i\cdot x_i+b)>=1\\
-Cy_i&&&y_i（x_i\cdot x_i+b)<1
\end{array}
\right.
\end{equation}
$$

按照梯度下降算法可以得到**线性SVM**的训练方法，仿照感知机的训练过程可知：

> 输入：训练集D，迭代次数M，惩罚因子C(注意这个C是超参数，需人为调参)，学习速率r
>
> 过程：
>
> 初始化参数
> $$
> w=(0，....,0)^T  ,b=0
> $$
> 对j=1,…..M:
>
> 1）算出误差向量$e=(e_1,…,e_N)^T$,其中
> $$
> e_i=1-y_i(w\cdot x_i+b)
> $$
> 分析一下：
>
> - 划分错误时 ei 后半部分减去一个负数 大于  1、
> - 划分正确时 ei 小于1，大于0，在几何上看在超平面Pi1和Pi2之间，表示离这两个平面的距离大小，该店为支持向量
> - 划分正确时ei 小于0，及调整参数时无需考虑该点，因为在Pi1和Pi2超平 面之外了
>
> 2）去除误差最大的一项：
> $$
> i=argmax(e_i)
> $$
> 3)若ei<=0,则找到这样一个超平面，推出循环体，否则去对应的样本来进行随机梯度下降
> $$
> w_{new}=w+(1-\eta)w+\eta Cy_ix_i
> $$
>
> $$
> b_new=b+\eta C y_i
> $$
>
> 输出： 线性SVM模型 $g(x)=sign(w \cdot x+b)$

可以看到线性的SVM模型于最后感知机的模型相同，只是这个线性SVM模型可以说时在无数个感知机模型中，划分效果最好的感知机模型。

> 总结：
>
> 其实线性SVM的原理是很容易理解的，但是在很多的资料书（包括周志华的西瓜书上）都直接将SVM作为一个普适的模型去理解，直接推导SVM的对偶形式，引入核函数，所以看起来十分的难以上手
>
> 对于线性SVM来说，回顾上面的推理，其核心就是找到感知机模型中划分效果最好的那个。也就是变成了一个优化所有点超平面几何距离的问题，并且我们定义了一些约束，将函数距离上的硬间隔变成了软间隔。将这些约束和优化问题相结合，最后用梯度下降法解决这一优化问题。



### 非线性SVM

> 该部分内容，只能浅度阐述
>
> - 实现非线性SVM的思路
> - 训练核SVM的方法（copy）
> - 核函数概念
>
> 因为后面对偶式的推导还未弄清楚



**---------------------------------------------------------SVM对偶式---------------------------------------------------------**

SVM软间隔最大化对偶形式（推导还未弄明白，直接看结果,推导真的没看明白，大概这就是智商不够高吧）：
$$
max -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i,x_j)+\sum_{i=1}^N\alpha_i
$$
满足约束条件对i=1,…,N有：
$$
\begin{equation}  
             \begin{array}{**lr**}  
            \sum_{i=1}^N \alpha_iy_i=0 \\  
         	0<=\alpha_i<=C
             \end{array}  
\end{equation}
$$
对SVM的训练：SMO算法：

主要手段：每次迭代中，只关注于两个变量的优化问题，以期望在剋已接受的恶时间内得到一个较优解，解决方案是在循环体中不断针对两个变量构造二次规划，并通过求解出其解析解来优化原始的对偶问题。步骤如下：

> - 考察所有变量（a1,a2,a3,…,an)及对应的样本点满足KKT条件的情况（这个KKT条件情况）
>
> - 若所有变量及对应样本点在容许误差内都满足KKT条件，则推出循环，完成训练。
>
> - 否则，通过如下步骤两个变量来构造新的规划问题：
>
>   - 选出违反KKT条件最严重的样本点，以其对应的变量作为第一个变量；
>   - 第二个变量的选取，可以随机选取。
>
> - 将上述步骤选出的变量以外的变量固定，仅仅针对两个变量进行优化。可推知此时问题转变为求二次规划的极大值。
>
> - 转化成二次规划的极大值后整个问题就变的十分容易，有约束条件和目标函数。构建拉格朗日方程，对其两个数值分别求偏导数为0，算得极值，利用梯度下降法更新两个变量的值。
>
>   
>
>   KKT条件（结论）：
>   $$
>   \begin{equation}  
>                \begin{array}{**lr**}  
>               	\alpha_i=0 \iff y_i g(x_i)>=1\\
>               	0<\alpha_i<C\iff y_i g(x_I)=1\\
>               	\alpha_i=C \iff y_ig(x_i)<=1
>                \end{array}  
>   \end{equation}
>   $$
>   违反KKT条件样本点的定义有多种，其中简单有效的定义：
>
>   计算损失向量$c=(c_1,c_2,…,c_n)^T$,其中
>   $$
>   c_i=[y_ig(x_i)-1]^2
>   $$
>   选取损失值最大的为变量：
>   $$
>   i=argmax\{e_i|i=1,2,...,N\}
>   $$





**------------------------------------------------------核技巧的概念-------------------------------------------------------**

**核技巧可以将线性算法升级为非线性算法**

可以理解为寻找一种映射，使得在低维度空间线性不可分的数据集，通过映射到高纬度的空间使其线性可分。

简单来说它通过核函数来避免显示定义映射，通过核函数
$$
K(x_i,x_y)=\phi(x_i)\cdot\phi(x_j)
$$
来替换算法中出现的内积
$$
x_i \cdot x_j
$$
来完成数据从低纬度映射到高纬度的过程。

整个过程思路如下：

- 将算法表述成样本点内积的组合（这也是为什么要化成对偶形式）
- 找到核函数k(xi,xj),能返回样本点映射后的内积
- 用k(xi,xj)，完成低维度到高纬度的映射（也就完成了，线性算法到非线性算法的转换）



下面列举所用的核函数：

![preview](../Machine%20Learning/Lab3%20SVM/assets/v2-63d3e66a8b7d273dac86bdcaddccc624_r.jpg)

其中 r ，b，d为人工设置的参数，d是一个正整数，r为正实数，b为非负实数都为超参数，需要单独设置

------

最后一般情况下将SVM的形式转换成对偶形式都是要利用核函数，来将线性模型提升为非线性模型。至于在核函数下SVM的训练方式，不在此赘述



> 总结一下
>
> 为什么引入对偶式？是因为要方便核函数的使用，因为对偶式里面存在内积。
>
> 对偶式和之前的训练方式不相同，它是通过SMO，每次选取两个变量，其他不变来逐步逼近最优解。
>
> 核函数在SVM中是一个十分重要的概念，在sklearn库中它也是可以进行选择的，它将线性模型转化成了非线性模型，是SVM的应用范围大大提高。



## SVM in sklearn

> 首先要说明的是sklearn中的SVM类是基于libsvm库进行编写的，底层的运算是C和cpython，因此可以加快其运行速度。详细了解SVM可以看libsvm的[相关文档](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
>
> 个人觉得理解好上文中的线性SVM对偶式训练方法SMO能够理解sklearn中参数的作用
>
> sklearn中的SVM都是用对偶式SMO进行训练的。
>
> 实验源码：[https://github.com/Zrealshadow/MachineLearning/tree/master/Lab3%20SVM](https://github.com/Zrealshadow/MachineLearning/tree/master/Lab3 SVM)



```python

#SVC：SVM分类模型
class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)

'''
-----------parameter--------

@param: C : float, optional (default=1.0)
错误的惩罚系数，在上文的推导中，惩罚系数越大，意味着模型对误分类点，或者不在间隔内的点的容忍越小，得出来的模型泛化在训练集上准确率越高。
@param: kernel : string, optional (default='rbf')
核函数的选择，在上文中介绍各个核函数的公式
	- "poly" 多项式
	- "linear" 线性
	- "rbf" 高斯核
	- "sigmoid"
通常情况下选用rbf的效果最好

@param: degree : int, optional (default=3)
选择“poly”核函数时需要指定的参数，多项式的次数，次数越多模型越复杂

@param: gamma : float, optional (default='auto') 
Current default is 'auto' which uses 1 / n_features
核函数上的超参数，对应上文的表为r，表示映射后数据的分布，gamma越大，支持向量越少，训练速度越快

@param: coef0 : float, optional (default=0.0)
在多项式核函数和sigmoid核函数中的b

@param:shrinking : boolean, optional (default=True)
预测哪些是支持向量，加速训练

@param: probability : boolean, optional (default=False)
输出每种概率的可能性

@param:tol : float, optional (default=1e-3)
停止训练的误差精度， SVM训练过程中使用的cv进行训练的，因此每一次更迭后都有 预测精度

@param:cache_size : float, optional
训练时可以缓存内存大小

@param: max_iter : int, optional (default=-1)
最大训练迭代次数，默认为-1，训练到停止为止

'''
#SVR：SVM回归模型

class sklearn.svm.SVR(kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

'''
其主要参数和SVC大致相同
'''

```



---



# Neural Network



## The Theory

> 最基础的神经网络，也是人们常说的神经网络通常指的是多层感知机，每一层的每一个节点，都会与下面一层所有的节点相连接，及全连接。在卷积神经网络中也称为全连接层。
>
> 在这种神经网络中，每一层的每一个节点（及每个神经元）都是一个感知机模型，每个神经元的输出通过激活函数（核函数）之后，输入到下一层神经元。在我的理解中，神经网络模型很大程度上接近于一个黑盒模型，因为说不清他为什么最后分类的效果会这么出色（可能是我不能理解）。在实践操作过程中，更多用经验调整参数，训练也是定义完损失函数，将输入数据规范化后即可。所以下面的文章中主要介绍神经网络如何训练（误差反向传播算法），损失函数的选择，激活函数的选择；



### Backpropagation（反向传播算法）

回顾之前所学的机器学习模型的训练步骤，无论是线性回归，SVM都被最后化成了一个优化问题，优化的目标就是损失函数，使损失函数最小。其实是可以把之前所说的感知机算法中的训练步骤拿来类比，因为神经网络本质上就是由多层多个感知机所组成。只不过其参数更多，训练过程更加的复杂，可能涉及多次求导。

下面以《机器学习》——周志华 书中的简单神经网络来介绍反向传播算法

![image-20190527103718447](../Machine%20Learning/LAB4%20MLP/assets/image-20190527103718447.png)

假设hidden层和输出层都用sigmoid函数。

网络对于训练样例$（x_k，y_k）$的损失函数（这里用均方误差）
$$
E_k=\frac{1}{2}\sum_{j=1}^l(y_j^k-\hat y_j^k)^2
$$
下面我们研究hidden层底h个节点对应j的参数$w_{hj}$,求偏导数
$$
w_{hj}^{new}=w_{hj}+\triangle w_{hj}
$$

$$
\triangle w_{hj}=-\eta \frac{\partial E_k}{\partial w_{hj}}
$$
通过**链式法则**，我们可以将 $ \frac{\partial E_k}{\partial w_{hj}}$拆分成易于求解的形式：
$$
\frac{\partial E_k}{\partial w_{hj}}=\frac{\partial E_k}{\partial \hat y_j^k}\cdot
 \frac{\partial \hat y_j^k}{\partial \beta_j}\cdot
 \frac{\partial \beta_j}{\partial w_{hj}}
$$
然后我们逐步求解

对于$\frac{\partial E_k}{\partial \hat y_j^k}$对上式的Ek进行求导可得：
$$
\frac{\partial E_k}{\partial \hat y_j^k}=y_j^k-\hat y_j^k
$$
对于$\frac{\partial \hat y_j^k}{\partial \beta_j}$，其实它是一个激活函数的求导，因为我们用的激活函数式sigmoid函数，其性质
$$
f_{'}(x)=f(x)(1-f(x))
$$
可知：
$$
\frac{\partial \hat y_j^k}{\partial \beta_j}=\hat {y_j^k}(1-\hat{y_j^k})
$$
对于$ \frac{\partial \beta_j}{\partial w_{hj}}$,我们根据图片右上角的式子进行求导
$$
\frac{\partial \beta_j}{\partial w_{hj}}=b_h
$$
因此
$$
\triangle w_{hj}=-\eta (y_j^k-\hat y_j^k)\hat {y_j^k}(1-\hat{y_j^k})b_h
$$
可以完成该节点的数据更新。

依次类推，如果是要求第三层节点的更新，通过链式法则：
$$
\frac{\partial E_k}{\partial v_{ih}}=\frac{\partial E_k}{\partial \hat y_j^k}\cdot
 \frac{\partial \hat y_j^k}{\partial \beta_j}\cdot
 \frac{\partial \beta_j}{\partial b_{h}}\cdot
 \frac{\partial b_h}{\partial \alpha_h} \cdot
 \frac{\partial \alpha_h}{\partial v_{ih}}
$$
逐一求导即可。该过程的详细细节可参考西瓜书。

> 因此，我们可以知道，在神经网络中参数更新的过程其实也和之前的机器学习的模型相同，按照梯度逼近最优解，只是其求导过程较为复杂，需要用到链式法则进行分步骤求导。
>
> 虽然在sklearn中MLP类对神经网络的训练过程进行了封装，但是了解其背后的算法原理总是有用的。



### 激活函数选择

直观来说，激活函数是整个结构中的**非线性扭曲力**，正是因为激活函数，神经网络可以划分非线性数据（例如抑或问题）下面根据发展时间介绍激活函数的种类。



**Sigmoid函数**

这个函数可以说很熟悉了，在logestic回归中我们就用到了这个函数，而且在SVM的kernal选择中也有这个核函数.它的公式在
$$
g(z)=\frac{1}{1+e^{-z}}
$$
![image-20190527151130481](../Machine%20Learning/LAB4%20MLP/assets/image-20190527151130481.png)

取值在(0，1)。适合输出为概率的情况

他的缺点：

- 梯度弥散，当神经元激活在接近0或者1时会饱和，在这些区域梯度几乎为0（可以从图中得知），这就会导致在训练过程中，更新权重十分缓慢，几乎没有更新。
- sigmoid函数不是中心对称的。如果输入神经元的总是正数，那么关于W的梯度在反向传播的过程中将要么全是正数，要么全是负数，这将会导致梯度下降权重更新时出现z字型的下降。
- exp函数 指数计算较为复杂耗时



**tanh函数**
$$
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
![image-20190527151110369](../Machine%20Learning/LAB4%20MLP/assets/image-20190527151110369.png)

在Sigmoid激活函数的基础上解决了中心对称的问题，但是两个缺点还是没有解决：

- 梯度弥散
- exp函数计算开销巨大



**ReLU函数**

现在无论是在视觉还是在自然语言处理领域主流的激活函数
$$
f(x)=max(0,x)
$$
![image-20190527151050990](../Machine%20Learning/LAB4%20MLP/assets/image-20190527151050990.png)

优点：

- 异常简单的计算复杂度，求导为1
- 单侧抑制，在x<0,不激活神经元，更符合生物学上人类的神经元。
- 相对宽阔的兴奋边界，在x∞时依旧不会产生梯度弥散问题
- 稀疏激活性，x<0，会使一些神经元"死亡"，（这也是一个缺点）

缺点：

- reLU函数会造成神经元死亡，而这种死亡是不可逆转的，因此导致了数据多样化的丢失。下面的Leaky ReLU 稍稍解决了这些问题。



**Leaky ReLU**
$$
f(x)=max(ex,x)
$$
![image-20190527151025772](../Machine%20Learning/LAB4%20MLP/assets/image-20190527151025772.png)

其中e是一个很小的负值梯度，上文介绍ReLU函数的时候说过，ReLU函数的缺点，神经元不可逆转死亡，那么引入了Leaky ReLU就解决了这个问题，这样使得神经元在处于非激活状态的情况下，信息不会完全丢失。



> 这里只介绍了sklearn库里面可选的激活函数(无 Leaky ReLU),但其实还有很多激活函数，在深度学习库中例如Pytorch，tensorflow，激活函数是可以自己定义的。现在sigmoid函数在深度学习中已经很少见到了，基本上在很多论文中都是利用的ReLU一类的函数。激活函数的选择对于训练和预测的效率速度有很大的影响。



### 损失函数的选择

> 在上文中提到很多次，在机器学习的很多问题中，我们都可以把整个问题简化成一个优化问题，而优化的目标函数就是损失函数。
>
> 在用神经网络解决的大部分问题中，网络结构大部分是固定的（例如我可以用Resnet网络的结构原封不懂的去提取特征，或者只是对网络的结构进行一些fine tune 微调）。而主要工作在如何清洗出高质量的输入数据和改善损失函数上面（例如可以在损失函数后面加上一些正则项或者惩罚项，来对特定问题的结果进行调整。就像在SVM中的过程一样，加一些限制条件，使得模型更加精准）因此损失函数的选择和调整在用神经网络解决具体问题的过程中十分重要。
>
> 下面介绍两个基本的损失函数



**距离损失函数（最小误差标准 Minimum Squared Error,mse）**

这个损失函数在SVM，回归分析中都用到了，是非常常见的损失函数，度量预测值与真实值的欧式距离
$$
L(y,\hat y)=\frac{1}{2}(y-\hat y)^2
$$
直观意义非常明确：预测值与真值的距离越大，损失就越大，反之就越小。



**交叉熵损失函数（Cross Entropy）**

交叉熵是信息论中的一个概念，在统计学中交叉熵的表现形式和最大似然法很相似。具体原理也不深究了，它定义了在一个概率分布上。根据公式，可以看到 负号内部的就是**最大似然法**的表现形式。
$$
L(y,\hat y)=-[y ln \hat y+(1-y)ln(1-\hat y)]
$$
交叉熵损失函数在基础的深度学习模型熵用的十分普遍，实践效果也很好



**---------------------------------------------------------如何选择损失函数---------------------------------------------------------**

**损失函数通常要结合输出层的激活函数来进行选取**，在BP算法中我们可以知道，链式法则求导过程中，第一步计算的局部梯度，是由损失函数对模型输出的梯度和激活函数的导数相乘得到的。而观察损失函数可以知道，损失函数一般较为复杂，通过选择相应的激活函数，能与激活函数的导数相乘简化运算或者降低激活函数本身的缺点带来的影响。下面介绍几种较为常见的组合及其优缺点。



**Sigmoid+MSE**

在logistic regression中就是这样的组合。

这样的组合未能解决Sigmoid函数梯度弥散的问题，对于某些极大的输入，模型参数更新十分缓慢。



**Sigmoid+Cross Entropy**

这里Cross Entropy 的导数
$$
L^{'}(y,\hat y)=-[\frac{y}{\hat y}-\frac{1-y}{1-\hat y}]=-\frac{y-\hat y}{\hat y(1-\hat y)}
$$
联系sigmoid的导数
$$
f_{'}(y)=\hat {y}(1-\hat{y})
$$
sigmoid 梯度弥散的根本原因就是当 $\hat y$趋紧与0或者1时$f^{'}(y)$会无穷小，导致更新换缓慢。但是如果选择Cross Entropy 从数学上来说 梯度弥散的问题就被完美解决了。



**Softmax+Cross Entropy**

在经典深度学习模型LeNet中的最后一层就是softmax+Cross Entropy. 

Softmax模型感觉上不像是一个激活函数，更像一个向量变换。能把模型的输出向量，归一化成一个概率向量。在多分类问题中Softmax模型用了很多。softmax函数的导函数形式与sigmoid相同。





> 总结：
>
> 在机器学习实验中，其实可以把上述的算法作为一个黑盒子来进行操作。利用sklearn里面的自动调参的API对其进行暴力便利，但是这是对于少量的数据集。
>
> 对于工业界的应用或者大数据比赛来说，数据集往往巨大，如果有懂得神经网络训练的算法细节，对应不同的项目和需求，调整损失函数（加正则项，惩罚项之类的），可以大大减少训练的时间，提高模型的得分。





## Neural Network in Sklearn

> 首先需要说明的是，Sklearn库中的关于深度学习的API是非常有限的。它只提供了一个多层感知机的接口。
>
> 完整的自定义程度更高深度学习模型还需用tensorflow或者pytorch来进行实现
>
> 实验源码：[https://github.com/Zrealshadow/MachineLearning/tree/master/LAB4%20MLP](https://github.com/Zrealshadow/MachineLearning/tree/master/LAB4 MLP)

在sklearn中，提供了两类多层感知机，一种是有监督学习多层感知机，一种是无监督学习的多层感知机。下面主要介绍有监督多层感知机的API的调用方法，也是实验中所用到的方法

另外要补充的是，在sklearn中MLP的损失函数是无法自定义的，它给固定成了：
$$
Loss(\hat y,y,W)=\frac{1}{2}||\hat y-y||^2+\frac{\alpha}{2}||W||^2
$$
（此处的W是MLP网络结构中的参数）

这就是上文所说的在MSE损失函数的基础上加上一个W的L2正则项，限定了参数的大小，具体原因参考上文Ridge回归。

```python
# 多层感知机分类模型
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

'''
----------parameter-------------
@param:hidden_layer_sizes  tuple, length = n_layers - 2, default (100,)
神经网络的结构，输入一个元组，元组的第i个元素，代表网络中第i层的神经元的个数

@param: activation  {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
激活函数的选择，其中’identity'是 f(x)=x，其他的激活函数上文都提过

@param：solver {'lbfgs', 'sgd', 'adam'}, default 'adam'
梯度下降法的方式，上面三个选项都是梯度下降法的优化方法：
	- 'lbfgs'：quasi-Newton方法的优化器 
	- 'sgd'：随机梯度下降
	- 'adam'：随机梯度下降法的优化方法
	注：在很多数据集中‘adam'无论是训练的效率和准确度都是最好的
	在小数据集上，'lbfgs'的速度可能表现更好
	
@param：alpha float, optional, default 0.0001
正则化项的系数，系数越大，多层感知机对于参数的要求越严格

@param:batch_size  int, optional, default 'auto'
批处理的大小，一次处理几项数据，当设置为auto，batch_size=min(200,n_samples) 
一般设置成2的次方训练速度可以提高

@param:learning_rate  {'constant', 'invscaling', 'adaptive'}, default 'constant'
学习速率的更新状态：
	- 'constant'： 给予恒定的学习速率
	- 'incscaling'：随着训练时间t，不断降低的学习速率 
		 effective_learning_rate =learning_rate_init / pow(t, power_t) power_t 在后面参数设定
	- 'adaptive':只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损			耗或验证分数停止升高至少tol时，将当前学习率除以5. 
注：只有当solver是‘sdg'时可以使用

@param: learning_rate_init  double, optional, default 0.001
初始学习速率

@param:  power_t double, optional, default 0.5，
见上文 'incscaling' 方法的学习速率更新

@param:max_iter: int, optional, default 200
最大迭代次数

@param: shuffle : bool, optional, default True
每次迭代后是否打乱样本集

@param:  tol float, optional, default 1e-4
对损失函数提升的容忍度，如果再一次迭代中，损失函数下降的数值小于tol，停止训练

@param:verbose : bool, optional, default False
是否打印训练过程

@param:warm_start : bool, optional, default False
是否沿用之前模型中已经有的参数继续训练

@param: momentum : float, default 0.9 
在0-1 之间 ，随机梯度下降中的动量的参数

@param：early_stopping : bool, default False
是否提前停止，如果true，模型将拿出训练集的10%作为测试集来验证，如果正确率不再提高，停止训练

@param: validation_fraction : float, optional, default 0.1
如果提前停止，作为测试集的比例


---------attribute----------

@attr: class_  array or list of array of shape (n_classes,)
返回分类的种类

@attr:loss_ : float
返回最后的损失函数

@attr:n_iter:int
返回迭代次数

@attr:n_layers_ : int
返回神经网络的层数

@attr:out_activation_ : string
返回激活函数
'''

#MLP回归模型
class sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
'''
方法属性参数与分类模型相同，见上。
'''
```

