import os
import numpy as np
import log
import argparse
import pandas as pd
import math
import random
import warnings
warnings.filterwarnings('ignore')
#日志打印到屏幕并保存到文件
if not os.path.exists('result'):
    os.mkdir('./result')
logger = log.Logger(filename='result/info.log').logger

class Node:
    """创建一个树的节点"""
    def __init__(self,
                 data_index,
                 split_feature=None,
                 split_value=None,
                 is_leaf=False,
                 loss=None,
                 current_depth=None):
        """
        :param data_index:该节点的数据在全部数据集中的索引
        :param split_feature: 最佳分割特征
        :param split_value:最佳分割特征值
        :param is_leaf:是否为叶子节点
        :param loss: 分类损失
        :param current_depth: 当前节点所在树的深度
        """
        self.loss = loss
        self.split_feature = split_feature
        self.split_value = split_value
        self.data_index = data_index
        self.is_leaf = is_leaf  #是不是叶子节点
        self.predict_value = None
        self.left_child = None  #如果当前节点可继续划分，则分为左子树和右子树
        self.right_child = None
        self.current_depth = current_depth

    def update_predict_value(self, targets, sample_weight=None):
        self.predict_value = self.loss.update_leaf_values(targets, sample_weight=sample_weight)
        logger.info('>>>>>>>>>>>叶子节点预测值：%.3f'%self.predict_value)

    def get_predict_value(self, instance):
        """
        预测结果，采用递归的方法
        :param instance: 一个样本
        :return:
        """
        if self.is_leaf:
            #如果是叶子节点，直接获得叶子节点值
            return self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)

class Tree:
    """创建一棵树，从根节点开始创建"""
    def __init__(self, X, y, max_depth=3, min_samples_split=2, features_name=None, loss=None, sample_weight=None):
        """
        初始化树的参数
        :param data:
        :param max_depth:树的最大深度
        :param min_samples_split:最小划分数据量
        :param sampel_weight: 样本权重
        :param features:特征值
        :param loss:
        :return: 树结构和节点
        """
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_weight=np.ones(len(X)) if sample_weight is None else sample_weight
        self.features_name = features_name
        self.remain_index = np.array([i for i in range(len(X))])  #当前节点的样本在原始数据中的下标索引
        self.leaf_nodes = []
        self.root_node = self.build_tree(X, y, self.remain_index, depth=0, sample_weight=self.sample_weight)  #根节点

    def build_tree(self, X, y, remain_index, depth=0, sample_weight=None):
        """
        构建一棵树
        此处有三个树继续生长的条件：
        1: 深度没有到达最大, 树的深度假如是3， 意思是需要生长成3层, 那么这里的depth只能是0, 1，所以判断条件是 depth < self.max_depth - 1
        2: 点样本数 >= min_samples_split
        3: 此节点上的样本的 target_name 值不一样（如果值 一样说明已经划分得很好了，不需要再分）
        :param sample_weight:
        :param X: 原始样本特征值
        :param y: 原始样本标签值
        :param remain_index:当前数据在原始数据中的索引
        :param depth: 树的当前深度
        :return: 树节点
        """
        now_X, now_y = X[remain_index], y[remain_index]
        n_samples, n_features=now_X.shape[0], now_X.shape[1]
        if depth < self.max_depth and n_samples >= self.min_samples_split and len(np.unique(now_y)) > 1:
            se = None
            split_feature = None
            split_value = None
            left_index_of_now_data = None
            right_index_of_now_data = None
            left_weight_index_of_now_data=None
            right_weight_index_of_now_data=None
            logger.info('----------树的深度：%d----------' % depth)

            """遍历每一个特征，分别计算最优的切割点"""
            for i in range(n_features):
                #logger.info('----------划分特征：%s----------'%(self.features_name[i]))
                #对特征进行预排序,获得是特征值的下标索引
                presort=np.argsort(now_X[:,i])
                '''遍历每个特征值的每个样本值'''
                for j in range(n_samples):
                    # 尝试划分
                    left_index =presort[:j]
                    right_index =presort[j:]
                    left_se = calculate_se(y[left_index], sample_weight=sample_weight[left_index])
                    right_se = calculate_se(y[right_index], sample_weight=sample_weight[right_index])
                    sum_se = left_se + right_se

                    if se is None or sum_se < se:
                        split_feature = i
                        split_value=X[remain_index[right_index[0]],i]
                        se = sum_se
                        left_index_of_now_data = remain_index[left_index]
                        right_index_of_now_data = remain_index[right_index]
                        left_weight_index_of_now_data=left_index
                        right_weight_index_of_now_data=right_index
            logger.info('----------最佳划分特征：%s----------'%split_feature)
            logger.info('----------最佳划分值：%.3f----------'%split_value)

            #找到当前最优划分特征和划分值，对当前这个节点进行处理，产生子节点
            node = Node(remain_index, split_feature, split_value, current_depth=depth)

            logger.info('----------构建左子树----------')
            node.left_child = self.build_tree(X, y, left_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[left_weight_index_of_now_data])
            logger.info('----------构建右子树----------')
            node.right_child = self.build_tree(X, y, right_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[right_weight_index_of_now_data])
            return node
        else:
            #不满足分裂条件，停止分裂，该节点就是叶子节点
            logger.info('----------树的深度：%d----------' % depth)
            node = Node(remain_index, is_leaf=True, loss=self.loss, current_depth=depth)
            node.update_predict_value(y[remain_index], sample_weight=sample_weight)
            self.leaf_nodes.append(node)
            return node

class BinomialDeviance:
    """二分类"""
    def initialize_f_0(self, y):
        """初始化 F_0 二分类的损失函数是对数损失，初始值是正样本的个数与负样本个数的比值取对数"""
        pos = y.sum()   #正样本
        neg = len(y) - pos   #负样本

        f_0_val = math.log(pos / neg)
        return np.ones(len(y))*f_0_val

    def gradients_(self,y_pred, y_true):
        """一阶导数"""
        return 1 / (1 + np.exp(-y_pred)) - y_true

    def hessians_(self, y_pred, y_true=None):
        """二阶导数"""
        return np.exp(-y_pred) / ((1 + np.exp(-y_pred)) ** 2)

    def calculate_residual(self, y, current_pred_value, iter):
        """计算负梯度"""
        residual=-self.gradients_(current_pred_value, y)
        return residual

    def update_f_m(self, X, current_pred_value, trees, iter, learning_rate):
        """计算 当前时刻的预测值F_m ，"""
        for leaf_node in trees[iter].leaf_nodes:
            current_pred_value[leaf_node.data_index]+=learning_rate*leaf_node.predict_value
        return current_pred_value

    def update_leaf_values(self, targets, sample_weight=None):
        """更新叶子节点的预测值"""
        if len(targets)!=0:
            return (targets*sample_weight).sum()/sample_weight.sum()
        else:
            return 0.0

    def get_train_loss(self, y, f, iter,):
        """计算训练损失"""
        loss = -2.0 * ((y * f) - f.apply(lambda x: math.exp(1+x))).mean()
        logger.info(('第%d棵树: log-likelihood:%.4f' % (iter, loss)))

class BaseGradientBoosting():
    def __init__(self, loss, learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2):
        super().__init__()
        self.loss = loss  #损失
        self.learning_rate = learning_rate  #学习率
        self.n_estimators = n_estimators  #树的棵数
        self.max_depth = max_depth   #单个树最大深度
        self.min_samples_split = min_samples_split #单节点最小划分样本量
        self.features_name=None
        self.trees = {}
        self.f_0 = {}

    def fit(self, X, y, sample_weight=None):
        """
        训练树模型
        :param X: 特征值
        :param y: 样本label真实值
        :param sample_weight: 样本权重，默认为None
        :return:
        """
        #TODO 删除id和label，得到特征名称，这里的特征名称是加密的，直接用代号就可以了
        self.features_name = ['fea_'+str(i) for i in range(X.shape[1]) ]

        # 初始化 f_0(x),初始化的计算方式取决于损失函数，对于（分类）交叉熵损失--来说，初始化 f_0(x) 就是 log(正/负)；
        self.f_0 = self.loss.initialize_f_0(y)

        # 对 m = 1, 2, ..., M
        current_pred_value=self.f_0  #当前树的预测值
        for iter in range(1, self.n_estimators+1): #遍历所有的树节点
            # 计算负梯度--对于平方误差来说就是残差
            logger.info('-----------------------------构建第%d颗树-----------------------------' % iter)
            residual= self.loss.calculate_residual(y, current_pred_value, iter=iter) #计算负梯度
            self.trees[iter] = Tree(X, residual, self.max_depth, self.min_samples_split, self.features_name, self.loss)
            current_pred_value=self.loss.update_f_m(X, current_pred_value, self.trees, iter, self.learning_rate)

class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self,learning_rate=0.1,n_estimators=100,max_depth=3,min_samples_split=2):
        super().__init__(BinomialDeviance(),learning_rate,n_estimators,max_depth,min_samples_split)

    def predict(self, X, return_proba=False):
        f_0=self.f_0
        score= np.zeros(X.shape[0])
        for iter in range(1, self.n_estimators + 1):
             for k in range(X.shape[0]):
                 score[k]+=self.trees[iter].root_node.get_predict_value(X[k])

        # 相比于回归任务，分类任务需把要最后累加的结果Fm(x)转成概率
        predict_proba=1/(1+np.exp(-score))
        if return_proba:
            return predict_proba
        score[predict_proba>=0.5]=1
        score[predict_proba<0.5]=0
        predict_label=score
        return predict_label

def calculate_se(label,sample_weight=None):
    """计算平方误差 mse"""
    mean = (label*sample_weight).sum()/sample_weight.sum()
    se = 0
    for y in label:
        se += (y - mean) * (y - mean)
    return se

def load_data():
    #训练数据和测试数据的demo
    from sklearn.datasets import load_breast_cancer
    x,y=load_breast_cancer(return_X_y=True)
    # data=[[1, 5, 20, 0],
    #       [2, 21, 70, 1],
    #       [3, 7, 30, 1],
    #       [4, 30, 60, 0]]
    # x=np.array(data)[:,[1,2]]
    # y=np.array(data)[:,-1]

    return x,y

def run(args):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # 获取训练和测试数据
    X,y = load_data()
    x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=20191126,test_size=0.1)

    model = GradientBoostingBinaryClassifier(learning_rate=args.lr,
                                             n_estimators=args.trees,
                                             max_depth=args.depth)
    # 训练模型
    model.fit(x_train,y_train)
    # 模型预测
    y_pred=model.predict(x_test,return_proba=False)
    print(y_pred)
    acc=accuracy_score(y_test,y_pred)
    print(acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--trees', default=5, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=3, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=False, type=bool, help='whether to print the log on the console')
    parser.add_argument('--plot', default=True, type=bool, help='whether to plot the decision trees')
    args = parser.parse_args()
    run(args)