# Support Vector Machine

[TOC]

## The Theory

>报告中大部分理论推导部分参考：书籍：《Python与机器学习实战》——何宇健， 并结合自己的理解进行阐述。



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

>初始化参数：
>$$
>w=(0，....,0)^T  ,b=0
>$$
> 对j=1,…..M:
>$$
>E=\{(x_i,y_i)|y_i(w\cdot x_i +b)<=0\}
>$$
>E为误分类点的集合，若E=空集 退出循环
>
>否则，任取E中一个点，利用它进行更新参数
>$$
>w_{new}=w+\eta y_i x_i
>$$
>
>$$
>b_{new}=b+\eta y_i
>$$
>
>最后输出：感知机模型
>$$
>g(x)=sign(w\cdot x+b)
>$$



**-----------------------------------------------------------感知机算法对偶形式----------------------------------------------------------**

>对偶形式的表示在SVM的核函数的理解中有很大的作用
>
>在我看来，这种表现形式比较新颖，是另外一种理解机器学习的角度

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

>这一步骤后很多资料上都写的，不妨设d* 为1，但是并没有说清楚 为什么 要取1。
>
> 其实分析可以发现这里的d*的取值对该优化问题的解是没有任何影响的。
>$$
>d^*=d\cdot||w||
>$$
>当d* 变成k d*时,在超平面不变的情况下：
>$$
>kw\cdot x_i+kb=kd^*
>$$
>及w和b也会变成相应的kw，kb。此时d 和不等式的约束(只是乘了一个系数)都没有变化。所以对于优化问题没有影响。因此不妨设d*=1。



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

>输入：训练集D，迭代次数M，惩罚因子C(注意这个C是超参数，需人为调参)，学习速率r
>
>过程：
>
>初始化参数
>$$
>w=(0，....,0)^T  ,b=0
>$$
>对j=1,…..M:
>
>1）算出误差向量$e=(e_1,…,e_N)^T$,其中
>$$
>e_i=1-y_i(w\cdot x_i+b)
>$$
>分析一下：
>
>- 划分错误时 ei 后半部分减去一个负数 大于  1、
>- 划分正确时 ei 小于1，大于0，在几何上看在超平面Pi1和Pi2之间，表示离这两个平面的距离大小，该店为支持向量
>- 划分正确时ei 小于0，及调整参数时无需考虑该点，因为在Pi1和Pi2超平 面之外了
>
>2）去除误差最大的一项：
>$$
>i=argmax(e_i)
>$$
>3)若ei<=0,则找到这样一个超平面，推出循环体，否则去对应的样本来进行随机梯度下降
>$$
>w_{new}=w+(1-\eta)w+\eta Cy_ix_i
>$$
>
>$$
>b_new=b+\eta C y_i
>$$
>
>输出： 线性SVM模型 $g(x)=sign(w \cdot x+b)$

可以看到线性的SVM模型于最后感知机的模型相同，只是这个线性SVM模型可以说时在无数个感知机模型中，划分效果最好的感知机模型。

>总结：
>
>其实线性SVM的原理是很容易理解的，但是在很多的资料书（包括周志华的西瓜书上）都直接将SVM作为一个普适的模型去理解，直接推导SVM的对偶形式，引入核函数，所以看起来十分的难以上手
>
>对于线性SVM来说，回顾上面的推理，其核心就是找到感知机模型中划分效果最好的那个。也就是变成了一个优化所有点超平面几何距离的问题，并且我们定义了一些约束，将函数距离上的硬间隔变成了软间隔。将这些约束和优化问题相结合，最后用梯度下降法解决这一优化问题。



### 非线性SVM

>该部分内容，只能浅度阐述
>
>- 实现非线性SVM的思路
>- 训练核SVM的方法（copy）
>- 核函数概念
>
>因为后面对偶式的推导还未弄清楚



**---------------------------------------------------------SVM对偶式-------------------------------------------------------**

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

>- 考察所有变量（a1,a2,a3,…,an)及对应的样本点满足KKT条件的情况（这个KKT条件情况）
>
>- 若所有变量及对应样本点在容许误差内都满足KKT条件，则推出循环，完成训练。
>
>- 否则，通过如下步骤两个变量来构造新的规划问题：
>
>  - 选出违反KKT条件最严重的样本点，以其对应的变量作为第一个变量；
>  - 第二个变量的选取，可以随机选取。
>
>- 将上述步骤选出的变量以外的变量固定，仅仅针对两个变量进行优化。可推知此时问题转变为求二次规划的极大值。
>
>- 转化成二次规划的极大值后整个问题就变的十分容易，有约束条件和目标函数。构建拉格朗日方程，对其两个数值分别求偏导数为0，算得极值，利用梯度下降法更新两个变量的值。
>
>  
>
>  KKT条件（结论）：
>  $$
>  \begin{equation}  
>               \begin{array}{**lr**}  
>              	\alpha_i=0 \iff y_i g(x_i)>=1\\
>              	0<\alpha_i<C\iff y_i g(x_I)=1\\
>              	\alpha_i=C \iff y_ig(x_i)<=1
>               \end{array}  
>  \end{equation}
>  $$
>  违反KKT条件样本点的定义有多种，其中简单有效的定义：
>
>  计算损失向量$c=(c_1,c_2,…,c_n)^T$,其中
>  $$
>  c_i=[y_ig(x_i)-1]^2
>  $$
>  选取损失值最大的为变量：
>  $$
>  i=argmax\{e_i|i=1,2,...,N\}
>  $$



**------------------------------------------------------核技巧的概念---------------------------------------------------------**

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

![preview](../../Machine%20Learning/Lab3%20SVM/assets/v2-63d3e66a8b7d273dac86bdcaddccc624_r.jpg)

其中 r ，b，d为人工设置的参数，d是一个正整数，r为正实数，b为非负实数都为超参数，需要单独设置

---

最后一般情况下将SVM的形式转换成对偶形式都是要利用核函数，来将线性模型提升为非线性模型。至于在核函数下SVM的训练方式，不在此赘述



>总结一下
>
>为什么引入对偶式？是因为要方便核函数的使用，因为对偶式里面存在内积。
>
>对偶式和之前的训练方式不相同，它是通过SMO，每次选取两个变量，其他不变来逐步逼近最优解。
>
>核函数在SVM中是一个十分重要的概念，在sklearn库中它也是可以进行选择的，它将线性模型转化成了非线性模型，是SVM的应用范围大大提高。



## SVM in sklearn

>首先要说明的是sklearn中的SVM类是基于libsvm库进行编写的，底层的运算是C和cpython，因此可以加快其运行速度。详细了解SVM可以看libsvm的[相关文档](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
>
>个人觉得理解好上文中的线性SVM对偶式训练方法SMO能够理解sklearn中参数的作用
>
>sklearn中的SVM都是用对偶式SMO进行训练的。
>
>实验源码：[https://github.com/Zrealshadow/MachineLearning/tree/master/Lab3%20SVM](https://github.com/Zrealshadow/MachineLearning/tree/master/Lab3 SVM)



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

