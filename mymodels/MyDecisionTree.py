from copy import deepcopy

import numpy as np
import sklearn.tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Node(object):
    def __init__(self, ):
        self.sample_index = None  # 保存当前节点中对应样本在数据集中的索引
        self.values = None  # 保存每个类别的数量 e.g. [5,10] 表示当前节点中第0个类别有5个样本，第1个类别有10个样本
        self.feature_id = -1  # 保存当前节点对应划分特征的id
        self.features = None  # 记录当前节点可用的剩余划分特征 e.g. [0,2]表示第0,2个特征在当前节点之前还没有使用过
        self.n_samples = 0  # 保存当前节点对应的样本数量
        self.left_child = None  # 保存当前节点的左孩子
        self.right_child = None  # 保存当前节点的右孩子
        self.criterion_value = 0.  # 保存当前节点对应的基尼系数
        self.split_value = None  # 预测样本划分节点时，选择左右孩子的的特征判断取值
        self.n_leaf = 0  # 以当前节点为根节点时其叶子节点的个数
        self.leaf_costs = 0.  # 以当前节点为根节点时其所有叶子节点的损失和


def Cart():
    model = MyCART()
    model = sklearn.tree.DecisionTreeClassifier()
    return model


class MyCART(object):
    def __init__(self, min_samples_split=2,
                 epsilon=1e-3,
                 pruning=False,
                 random_state=None):
        self.root = None
        self.min_samples_split = min_samples_split  # 用来控制是否停止分裂
        self.epsilon = epsilon  # 停止标准
        self.pruning = pruning  # 是否需要进行剪枝
        self.random_state = random_state

    def _compute_gini(self, y_class):
        """
        计算基尼指数
        :param y_class:  np.array   [n,]
        :return:
        """
        y_unique = np.unique(y_class)
        if y_unique.shape[0] == 1:  # 只有一个类别
            return 0.  # 基尼指数为0
        gini = 0.
        for i in range(len(y_unique)):  # 取每个类别
            p = np.sum(y_class == y_unique[i]) / len(y_class)
            gini += p ** 2
        gini = 1 - gini
        return gini

    def _compute_gini_da(self, f_id, data):
        """
        输入特征列索引f_id以及样本数据，计算得到这一列特征中不同特征取值下的基尼指数
        :param f_id:
        :param data:
        :return: 当前特征维度下，不同特征取值时的最小基尼指数, 离散特征区间的起始索引，以及对应的样本划分索引
        e.g.
        x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                      [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
        dt = MyCART()
        dt.feature_values = dt._get_feature_values(x)
        dt._y = y
        print(dt.feature_values)
        # {0: [0.5], 1: [0.5], 2: [0.5, 1.5]}
        X = np.hstack(([x, np.arange(len(x)).reshape(-1, 1)]))
        r = dt._compute_gini_da(0, X)
        print(r)
        (0.3452380952380953, 0, array([ True,  True,  True,  True,  True,  True,  True, False, False,
               False, False, False, False, False, False]))
        """
        feature_values = self.feature_values[f_id]  # 取当前f_id列特征对应的离散化取值情况
        x_feature = data[:, f_id]  # 取f_id对应的特征列
        x_ids = np.array(data[:, -1], dtype=np.int64).reshape(-1)  # 样本索引
        labels = self._y[x_ids]
        min_gini = 99999.
        split_id = None
        split_sample_idx = None
        for i in range(len(feature_values)):  # 遍历当前特征维度中，离散化特征的每个取值
            index = (x_feature <= feature_values[i])
            # 判断特征的取值是否 <= 特征分裂值（即左孩子对应的索引），并以此将当前节点中的样本划分为左右两个部分
            if np.sum(index) < 1.:  # 如果当前特征取值范围没有样本，则继续
                continue
            d1, y1 = data[index], labels[index]  # 根据当前特征维度的取值将样本划分为两个部分，左子树
            d2, y2 = data[~index], labels[~index]  # 右子树
            gini = len(y1) / len(index) * self._compute_gini(y1) + \
                   len(y2) / len(index) * self._compute_gini(y2)
            if gini < min_gini:  # 保存当前特征维度下，能使基尼指数最小时的特征取值
                min_gini = gini
                split_id = i
                split_sample_idx = index
        return min_gini, split_id, split_sample_idx

    def _get_feature_values(self, data):
        n_features = data.shape[1]
        feature_values = {}
        for i in range(n_features):
            x_feature = sorted(set(data[:, i]))  # 去重与排序
            tmp_values = []
            for j in range(1, len(x_feature)):
                tmp_values.append(round((x_feature[j - 1] + x_feature[j]) / 2, 4))  # 计算均值
            feature_values[i] = tmp_values
        return feature_values

    def _build_tree(self, data, f_ids):
        print("f_ids: ", f_ids)
        x_ids = np.array(data[:, -1], dtype=np.int64).reshape(-1)  # 取每个样本在数据集中对应的索引
        node = Node()
        node.sample_index = x_ids  # 当前节点所有样本的索引
        labels = self._y[x_ids]  # 当前节点所有样本对应的标签
        node.n_samples = len(labels)  # 当前节点的样本数量
        node.values = np.bincount(labels, minlength=self.n_classes)  # 当前节点每个类别的样本数
        node.features = f_ids  # 当前节点状态时特征集中剩余特征
        if self.root is None:
            self.root = node

        y_unique = np.unique(labels)  # 当前节点中存在的类别情况
        if y_unique.shape[0] == 1 or len(f_ids) < 1 \
                or node.n_samples <= self.min_samples_split:  # 只有一个类别或特征集为空或样本数量少于min_samples_split
            return node
        gini = self._compute_gini(labels)  # 计算当前节点对应的基尼指数
        node.criterion_value = gini
        if gini < self.epsilon:
            return node
        min_gini = 99999
        split_id = None  # 保存所有可用划分特征中，能够值得基尼指数最小的特征 对应特征离散区间的起始索引
        split_sample_idx = None  # 最小基尼指数下对应的样本划分索引
        best_feature_id = -1  # 保存所有可用划分特征中，能够使得基尼指数最小的特征 对应的特征ID
        for f_id in f_ids:  # 遍历每个特征
            # 遍历特征下的每种取值方式的基尼指数，并返回最小的
            m_gini, s_id, s_s_idx = self._compute_gini_da(f_id, data)
            if m_gini < min_gini:  # 查找所有特征所有取值方式下，基尼指数最小的
                min_gini = m_gini
                split_id = s_id
                split_sample_idx = s_s_idx
                best_feature_id = f_id
        print('best_feature_id: ', best_feature_id)
        node.feature_id = best_feature_id
        feature_values = self.feature_values[best_feature_id]
        node.split_value = feature_values[split_id]
        left_data = data[split_sample_idx]
        right_data = data[~split_sample_idx]
        candidate_ids = deepcopy(f_ids)
        candidate_ids.remove(best_feature_id)  # 当前节点划分后的剩余特征集，同一个子树中特征只会用到一次。
        if len(left_data) > 0:
            node.left_child = self._build_tree(left_data, candidate_ids)  # 递归构建决策树
        if len(right_data) > 0:
            node.right_child = self._build_tree(right_data, candidate_ids)
        return node

    def fit(self, X, y):
        """
        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        if self.pruning:  # 如果剪枝则划分一部分数据作为测试集
            X, self.x_test, y, self.y_test = train_test_split(X, y,
                                                              test_size=0.1,
                                                              random_state=self.random_state)
        self._y = np.array(y).reshape(-1)
        self.n_classes = len(np.bincount(y))  # 得到当前数据集的类别数量
        feature_ids = [i for i in range(X.shape[1])]  # 得到特征的序号
        self.feature_values = self._get_feature_values(X)  # 得到离散化特征
        self._X = np.hstack(([X, np.arange(len(X)).reshape(-1, 1)]))
        # 将训练集中每个样本的序号加入到X的最后一列
        self._build_tree(self._X, feature_ids)  # 递归构建决策树
        if self.pruning:  # 进行剪枝
            self._pruning_leaf()

    def level_order(self, return_node=False):
        """
        层次遍历
        :return:
        """
        root = self.root
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            tmp = []
            for i in range(len(queue)):
                node = queue.pop(0)
                tmp.append(node)
                if node.left_child:
                    queue.append(node.left_child)
                if node.right_child:
                    queue.append(node.right_child)
            res.append(tmp)
        if return_node:
            return res  # 按层次遍历的顺序返回各层节点的地址
            # [[root], [level2 node1, level2_node2], [level3,...] [level4,...],...[],]

    def _predict_one_sample(self, x):
        """
        预测单一样本
        :param x: [n_features,]
        :return:
        """
        current_node = self.root
        while True:
            # 有些情况下叶子节点没有兄弟节点
            if not current_node.left_child or \
                    not current_node.right_child or \
                    current_node.split_value is None \
                    or current_node.n_samples < self.min_samples_split:
                # 当前节点为叶子节点
                return current_node.values
            current_feature_id = current_node.feature_id
            current_feature = x[current_feature_id]
            split_value = current_node.split_value
            if current_feature <= split_value:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

    def predict(self, X):
        """
        :param X: shape [n_samples,n_features]
        :return:
        """
        results = []
        for x in X:
            results.append(self._predict_one_sample(x))
        results = np.array(results)
        y_pred = np.argmax(results, axis=1)
        return y_pred

    def _get_pruning_gt(self, node):
        """
        计算对当前节点剪枝前和剪枝后对应的gt值
        :return:
        """

        def _compute_cost_in_leaf(labels):
            """
            计算节点的损失  c = -\sum_{k=1}^KN_{tk}\log{\frac{N_{tk}}{N_t}}
            :param labels:
            :return:
            e.g. y_labels = np.array([1, 1, 1, 0])
            _compute_cost_in_leaf(y_labels)   3.24511
            """
            y_count = np.bincount(labels)
            n_samples = len(labels)
            cost = 0
            for i in range(len(y_count)):
                if y_count[i] == 0:
                    continue
                cost += y_count[i] * np.log2(y_count[i] / n_samples)
            return -cost

        if not node.left_child and not node.right_child:
            node.leaf_costs = _compute_cost_in_leaf(self._y[node.sample_index])
            # 如果当前节点是叶子节点，则计算该叶子节点对应的损失值
            return 99999.
        parent_cost = _compute_cost_in_leaf(self._y[node.sample_index])  # 计算以当前节点为根节点剪枝后的损失
        if node.left_child:
            node.leaf_costs += node.left_child.leaf_costs  # 以当前节点为根节点累计剪枝前所有叶子节点的损失
        if node.right_child:
            node.leaf_costs += node.right_child.leaf_costs
        g_t = (parent_cost - node.leaf_costs) / (node.n_leaf - 1 + 1e-5)  # 计算gt，其中1e-5为平滑项
        return g_t

    def _get_subtree_sequence(self):
        """
        本方法的作用是得到决策树中所有的子树序列
        :return:
        """
        subtrees = []
        stop = False
        count_t = 0
        while not stop:
            if not self.root.right_child and not self.root.left_child:
                stop = True
            # while self.root.right_child and self.root.left_child:
            level_order_nodes = self.level_order(return_node=True)
            best_gt = 99999.
            best_pruning_node = None
            for i in range(len(level_order_nodes) - 1, -1, -1):  # ******** 从对底层向上遍历 ******
                current_level_nodes = level_order_nodes[i]  # 取第i层的所有节点
                for j in range(len(current_level_nodes)):  # ******  从左向右遍历 *********
                    current_node = current_level_nodes[j]  # 取第i层的第j个节点
                    current_node.n_leaf = 0  # 对于每一颗子树来说，重置计数，因为原始值中包含有上一课子树的计数信息
                    current_node.leaf_costs = 0.  # 因为需要在每一颗子树中保存相关信息
                    if current_node.left_child is not None:
                        current_node.n_leaf += current_node.left_child.n_leaf  # 计算以当前节点为根节点的叶子节点数
                    if current_node.right_child is not None:
                        current_node.n_leaf += current_node.right_child.n_leaf
                    elif not current_node.left_child and not current_node.right_child:
                        current_node.n_leaf = 1  # 当前节点为叶子节点，则其对应的叶子节点数为1
                    gt = self._get_pruning_gt(current_node)
                    if gt < best_gt:
                        best_gt = gt
                        best_pruning_node = current_node
            count_t += 1
            subtrees.append(deepcopy(self.root))
            if not stop:
                best_pruning_node.left_child = None
                best_pruning_node.right_child = None  # 剪枝
        return subtrees

    def _pruning_leaf(self):
        subtrees = self._get_subtree_sequence()  # 得到所有子树序列T0,T1,T2,...,Tn
        best_tree = None
        max_acc = 0.
        for tree in subtrees:  # 在测试集上对所有子树进行测试，
            self.root = tree  # 选择准确率最高的子树作为最终的决策树
            acc = accuracy_score(self.predict(self.x_test), self.y_test)
            if acc > max_acc:
                max_acc = acc
                best_tree = tree
        self.root = best_tree


def load_simple_data():
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    return x, y


def count_children():
    x, y = load_simple_data()
    dt = MyCART(min_samples_split=2)
    dt.fit(x, y)
    level_order_nodes = dt.level_order(return_node=True)
    for i in range(len(level_order_nodes) - 1, -1, -1):
        current_level_nodes = level_order_nodes[i]  # 取第i层的所有节点
        for j in range(len(current_level_nodes)):
            current_node = current_level_nodes[j]  # 取第i层的第j个节点
            if current_node.left_child is not None:
                current_node.n_leaf += (current_node.left_child.n_leaf)
            if current_node.right_child is not None:
                current_node.n_leaf += (current_node.right_child.n_leaf)
            if not current_node.left_child and not current_node.right_child:
                current_node.n_leaf = 1

    dt.level_order()
