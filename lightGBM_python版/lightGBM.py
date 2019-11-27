import os
import numpy as np
import log
import argparse
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
    def __init__(self,
                 X,
                 y,
                 current_tree,
                 max_depth=3,
                 min_samples_split=2,
                 features_name=None,
                 loss=None,
                 sample_weight=None,
                 max_bin=255,
                 min_data_bin=1):
        """
        初始化树的参数
        :param X:
        :param y:
        :param max_depth:树的最大深度
        :param min_samples_split:最小划分数据量
        :param sample_weight: 样本权重
        :param features_name:特征名称
        :param loss:
        :param max_bin:桶的最大数量
        :param min_data_bin:桶内最小样本量，目前测试阶段设置为1
        :return: 树结构和节点
        """
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_weight=np.ones(len(X)) if sample_weight is None else sample_weight
        self.features_name = features_name
        self.remain_index = np.array([i for i in range(len(X))])  #当前节点的样本在原始数据中的下标索引
        self.leaf_nodes = []
        self.max_bin=max_bin
        self.min_data_bin=min_data_bin
        self.root_node = self.build_tree(X, y, current_tree, self.remain_index, depth=0, sample_weight=self.sample_weight)  #根节点

    def build_tree(self, X, y, current_tree, remain_index, depth=0, sample_weight=None):
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
        now_X, now_y, now_current_tree = X[remain_index], y[remain_index], current_tree[remain_index]
        n_samples, n_features=now_X.shape[0], now_X.shape[1]

        if depth < self.max_depth and n_samples >= self.min_samples_split and len(np.unique(now_y)) > 1:
            split_feature = None
            split_value = None
            left_index_of_now_data = None
            right_index_of_now_data = None
            left_weight_index_of_now_data=None
            right_weight_index_of_now_data=None

            # 确定分桶大小
            # max_bin = min(self.max_bin, int(n_samples / self.min_data_bin))  # 如果每个桶的数据量都是最少的情况下，分桶的数量小于给定的桶数，那么取前者，也就是取两者的最小值
            maxInfoGain = -np.inf  # 初始化信息增益
            # 当前叶子节点的一阶导数总和
            G_all = self.loss.gradients_(y_pred=now_current_tree, y_true=now_y).sum()
            # 当前叶子节点的样本数量
            n_all = n_samples
            logger.info('----------树的深度：%d----------' % depth)

            """遍历每一个特征，分别计算最优的切割点"""
            for i in range(n_features):
                #logger.info('----------划分特征：%s----------'%(self.features_name[i]))
                #对特征进行分桶
                H = {}  # 构建一个直方图
                for j in range(n_samples):
                    H[now_X[j][i]] = H.get(now_X[j][i], [0, 0])  # 存储当前桶的一阶梯度和以及样本数量
                    # 计算当前叶子节点的一阶导数值
                    H[now_X[j][i]][0] += self.loss.gradients_(now_current_tree[j], now_y[j])
                    H[now_X[j][i]][1] += 1
                '''遍历每个特征值的每个样本值'''
                # 选择前max_bin个bin，并将小于min_data_num的bin删掉
                binSet = sorted(H.items(), key=lambda x: x[1][0], reverse=True)
                binSet = [bin for bin in binSet if bin[1][1] >= self.min_data_bin][:self.max_bin]
                # 数据已经分完桶，开始计算信息增益
                sl, nl = 0, 0  # 当前桶左边的梯度之和与样本数量
                for k in range(len(binSet)):
                    sl += binSet[k][1][0]
                    nl += binSet[k][1][1]
                    sr = G_all - sl
                    nr = n_all - nl
                    # 计算当前节点的信息增益
                    infoGain = sl ** 2 / nl + sr ** 2 / nr - G_all ** 2 / n_all
                    if infoGain > maxInfoGain:
                        maxInfoGain=infoGain
                        split_feature = i
                        split_value = binSet[k][0]
                        left_index=list(now_X[:,i]<split_value)
                        right_index=list(now_X[:,i]>=split_value)
                        left_index_of_now_data=remain_index[left_index]
                        right_index_of_now_data=remain_index[right_index]
                        left_weight_index_of_now_data=left_index
                        right_weight_index_of_now_data=right_index
            logger.info('----------最佳划分特征：%s----------'%split_feature)
            logger.info('----------最佳划分值：%.3f----------'%split_value)

            #找到当前最优划分特征和划分值，对当前这个节点进行处理，产生子节点
            node = Node(remain_index, split_feature, split_value, current_depth=depth)

            logger.info('----------构建左子树----------')
            node.left_child = self.build_tree(X, y, current_tree, left_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[left_weight_index_of_now_data])
            logger.info('----------构建右子树----------')
            node.right_child = self.build_tree(X, y, current_tree, right_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[right_weight_index_of_now_data])
            return node
        else:
            #不满足分裂条件，停止分裂，该节点就是叶子节点
            logger.info('----------树的深度：%d----------' % depth)
            node = Node(remain_index, is_leaf=True, loss=self.loss, current_depth=depth)
            node.update_predict_value(current_tree[remain_index], sample_weight=sample_weight)
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
        self.bundles=None

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
        # current_pred_value=self.f_0  #当前树的预测值
        current_pred_value=np.zeros(X.shape[0])  #当前树的预测值
        sample_data_index=[i for i in range(X.shape[0])]
        logger.info('----------开始互斥特征合并----------')
        X, y = self.efb(X, y)
        for iter in range(1, self.n_estimators+1): #遍历所有的树节点
            logger.info('-----------------------------构建第%d颗树-----------------------------' % iter)
            logger.info('----------开始单边梯度采样----------')
            sample_data_index,sample_weight=self.goss(X[sample_data_index],y_true=y[sample_data_index],y_pred=current_pred_value)
            print(sample_data_index)
            if len(sample_data_index)==0:
                logger.info("----------单边采样数据量为0，训练结束----------")
                break

            # 计算负梯度
            # residual= self.loss.calculate_residual(y[sample_data_index], current_pred_value[sample_data_index], iter=iter) #计算负梯度
            self.trees[iter] = Tree(X[sample_data_index], y[sample_data_index], current_pred_value[sample_data_index], self.max_depth, self.min_samples_split, self.features_name, self.loss, sample_weight=sample_weight)
            current_pred_value=self.loss.update_f_m(X[sample_data_index], current_pred_value[sample_data_index], self.trees, iter, self.learning_rate)

    def goss(self, X, y_true, y_pred, alpha=0.2, beta=0.1, random_state=20191127):
        """
        单边梯度采样
        :param X:
        :param y_true:lable 真实值
        :param y_pred:label 预测值
        :param alpha: 大梯度样本采样比例
        :param beta: 小梯度样本采样比例
        :param random_state: 随机种子
        :return:新数据以及样本权重
        """
        random.seed(random_state)
        # 计算所有样本的一阶导数和二阶导数，用的是对数损失函数
        n_samples=X.shape[0]
        grad_ = self.loss.gradients_(y_pred=y_pred,y_true=y_true)
        hess_ = self.loss.hessians_(y_pred=y_pred,y_true=y_true)
        # 对梯度进行排序
        gradientIndexSort = np.argsort(abs(grad_*hess_))  # 升序排列
        # 大梯度样本数量
        maxGrad = int(n_samples * alpha)
        # 因为是升序排列，所以取最后的maxGrad个样本
        num = n_samples - maxGrad
        topSet = gradientIndexSort[num:]  # 大梯度的值,这一部分的权重是不需要改变的
        # 小梯度样本，需要从剩下的样本中随机抽取b占比的样本吗，并修改这些样本的权重
        randSet = random.sample(list(gradientIndexSort[:num]), k=int(n_samples * beta),)
        # 新的数据集合
        indexSort = np.concatenate([topSet, randSet], axis=0)
        # 更新权重值
        sample_weight = np.ones(indexSort.shape[0])
        sample_weight[int(n_samples * alpha):] *= (1 - alpha) / beta
        return indexSort, sample_weight

    def efb(self, X, y=None):
        """
        互斥特征绑定 Exclusive Feature Bundling
        1，如何判断两个特征互斥：根据特征中的非0值判断两个特征的冲突度是多少,非零值越大，冲突越大
        2，两个互斥的特征怎么绑定：
        :param X:原始x
        :param y:原始y，虽然不对y进行处理
        :return:处理过后的新的数据集
        """
        # 计算样本特征中的非零值的个数
        n_samples, n_features = X.shape[0], X.shape[1]
        Xcopy = X.copy()
        Xcopy[Xcopy != 0] = 1
        nonzeroCounts = np.sum(Xcopy, axis=0)  # [1,0,2]  每个特征非零值的数量
        feature_index = np.argsort(nonzeroCounts)  # 对这个数量进行一个排序，获得特征的索引，后面要用这个索引进行特征的合并
        max_conflict_count = nonzeroCounts.sum() / len(nonzeroCounts)  # 用这些非零值的均值作为最大的冲突量
        bundles, bundle = [], {'index': [], 'conflict_count': 0}  # 前一个是所有捆的集合，后一个是放所有互斥的特征
        for i in range(len(feature_index)):
            index = feature_index[i]  # 当前特征索引，告诉我们是那个特征，是根据非零值从小到大排序的
            current_nonzeroCount = nonzeroCounts[index]  # 当前特征的非零值的个数
            if len(bundle['index']) == 0 and len(bundles) == 0:
                bundle['index'].append(index)
                bundle['conflict_count'] += current_nonzeroCount
                bundles.append(bundle)
            else:
                if bundles[-1]['conflict_count'] + current_nonzeroCount <= max_conflict_count:
                    # 如果加在现成的捆中总冲突量小于最大冲突量，则可以加到这个捆中
                    bundles[-1]['index'].append(index)
                    bundles[-1]['conflict_count'] += current_nonzeroCount
                else:
                    # 如果大于最大冲突量了，则新建一个捆
                    bundle = {'index': [], 'conflict_count': 0}
                    bundle['index'].append(index)
                    bundle['conflict_count'] += current_nonzeroCount
                    bundles.append(bundle)
        self.bundles=bundles #这个桶保存下来，预测的时候需要用到

        logger.info('----------互斥绑定后特征数量为%d----------' % len(bundles))
        # 现在找完互斥特征了，下面是把他们合并
        newSet = []
        for bundle in bundles:
            ef_index = bundle['index']  # 可以捆绑的特征
            sub_x = X[:, ef_index]
            max_value_per_fea = np.max(sub_x, axis=0)  # 每个特征的最大值，找一个偏移量
            offset = max_value_per_fea[0]
            if len(ef_index) > 1:
                a = sub_x[:, ef_index[1:]]
                a[a != 0] += offset
                new_fea = np.add(a, sub_x[:, [ef_index[0]]])
                newSet.append(new_fea)
            else:
                newSet.append(sub_x)
        new_X = np.concatenate(newSet, axis=1)  # 生成新的特征
        if y is not None:
            new_y=y
            return new_X, new_y
        return new_X

class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self,learning_rate=0.1,n_estimators=100,max_depth=3,min_samples_split=2):
        super().__init__(BinomialDeviance(),learning_rate,n_estimators,max_depth,min_samples_split)

    def predict(self, X, return_proba=False):
        f_0=self.f_0
        #训练的时候进行了特征捆绑，特征数量少了，但是测试集的特征没有少，所以也要对测试集进行特征绑定
        newSet=[]
        for bundle in self.bundles:
            if len(bundle['index'])>1:
                a=np.sum(X[:,bundle['index']],axis=1)
                newSet.append(a)
            else:
                newSet.append(X[:,bundle['index']])
        X=np.concatenate(newSet,axis=1)
        score= np.zeros(X.shape[0])
        for tree in self.trees.values():
            for k in range(X.shape[0]):
                score[k]+=tree.root_node.get_predict_value(X[k])

        # 相比于回归任务，分类任务需把要最后累加的结果Fm(x)转成概率
        predict_proba=1/(1+np.exp(-score))
        if return_proba:
            return predict_proba
        score[predict_proba>=0.5]=1
        score[predict_proba<0.5]=0
        predict_label=score
        return predict_label


def calculate_se(label,sample_weight=None):
    """计算平方误差"""
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
    x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=20191126,test_size=0.3)

    # 创建模型结果的目录
    # 初始化模型
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