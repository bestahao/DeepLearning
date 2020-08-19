import os, collections, random, math
from heapq import *

# 霍夫曼树的节点类
class Node:
    def __init__(self, id, num):
        self.parent = None
        self.id = id
        self.num = num

    def __lt__(self, other):
        return self.num < other.num


# 封装一个类来做数据预处理
class DataProcess:
    def __init__(self):
        self.model_save = {}

    # prefix: '../../../data/', file: 'ptb.train.txt'
    def tokenize(self, prefix, file):
        assert file in os.listdir(prefix)

        with open(prefix + file, 'r') as f:
            # 句子以换行符为分割
            lines = f.readlines()
            # st是sentence的缩写，单词以空格为分割
            raw_dataset = [st.split() for st in lines]

        # sentences: 42068
        print('# sentences: %d' % len(raw_dataset))

        # 简单看下语料的情况
        for st in raw_dataset[:3]:
            print('# tokens:', len(st), st[:5])

        # 建议词语索引
        counter = collections.Counter([token for sentence in raw_dataset for token in sentence])
        counter = dict(filter(lambda x:x[1], counter.items()))
        self.model_save['counter'] = counter

        # 建立词映射到整数索引
        id_to_token = [token for token, _ in counter.items()]
        token_to_id = {tk: id for id, tk in enumerate(id_to_token)}
        dataset = [[token_to_id[tk] for tk in st if tk in token_to_id]
                   for st in raw_dataset]
        self.model_save['id_to_token'] = id_to_token
        self.model_save['vocabulary_size'] = len(id_to_token)
        self.model_save['token_to_id'] = token_to_id
        self.model_save['id_dataset'] = dataset

        # 计算token总数方便后续的二次采样
        num_tokens = sum([len(st) for st in dataset])
        self.model_save['num_tokens'] = num_tokens

        # 二次采样
        # 比起窗口中总是出现一些高频词（一些无具体意义的词），我们希望出现一些真正有价值，但是出现频率不高的词
        # 通过二次采样，丢弃一些词；频率越高的词，被丢弃的几率越大
        # 丢弃并不是这一词的被整个丢弃，只是减少了出现的次数

    def sub_sample(self):
        # 确定被丢弃的概率
        def discard(idx):
            counter = self.model_save['counter']
            id_to_token = self.model_save['id_to_token']
            return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / counter[id_to_token[idx]] * self.model_save['num_tokens']

            )

        id_dataset = self.model_save['id_dataset']
        subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in id_dataset]
        self.model_save['subsampled_dataset'] = subsampled_dataset
        counter = collections.Counter([token for sentence in subsampled_dataset for token in sentence])
        self.model_save['counter'] = counter

    # 它每次在整数1和 max_window_size （最⼤背景窗⼝）之间随机均匀采样⼀个整数作为背景窗⼝⼤⼩
    def get_centers_and_contexts(self, max_window_size):
        centers, contexts = [], []
        dataset = self.model_save['subsampled_dataset']
        for sentence in dataset:
            # 至少需要两个词，才能组成一对中心词与背景词
            if len(sentence) < 2:
                continue

            centers += sentence
            for center_i in range(len(sentence)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i-window_size),
                                     min(len(sentence), center_i+window_size+1)))
                indices.remove(center_i)
                contexts.append([sentence[idx] for idx in indices])

        return centers, contexts

    # 接下来是对负采样的实现，负采样是近似训练的一种方法
    # 我们使⽤负采样来进⾏近似训练。对于⼀对中⼼词和背景词，我们随机采样K个噪声词（实验中设K=5）
    # 根据word2vec论⽂的建议，噪声词采样概率P(w)设为w词频与总词频之⽐的0.75次⽅

    def get_negatives(self, sampling_weights, K, max_window_size=5):
        all_centers, all_contexts = self.get_centers_and_contexts(max_window_size)

        all_negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for contexts in all_contexts:
            negatives = []
            # 每个背景词都要有K个噪声词
            while len(negatives) < len(contexts) * K:
                if i == len(neg_candidates):
                    # 根据每个词的权重（sampling_weights）随机⽣成k个词的索引作为噪声词。
                    # 为了⾼效计算，可以将k设得稍⼤⼀点
                    # 返回包含k个的，按照权重随机取
                    i, neg_candidates = 0, random.choices(
                        population, sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i + 1
                # 噪声词不能是背景词
                # 如果是背景词就略过
                if neg not in set(contexts):
                    negatives.append(neg)

            all_negatives.append(negatives)
        self.model_save['all_centers'] = all_centers
        self.model_save['all_contexts'] = all_contexts
        self.model_save['all_negatives'] = all_negatives

    def huffle_tree(self):
        counter = self.model_save['counter']

        id_to_token = self.model_save['id_to_token']
        word_nodes = [None] * len(id_to_token)

        # 根据词频建立最小堆，再依次建立霍夫曼树
        node_heap = []
        for key, value in counter.items():
            node = Node(0, value)
            word_nodes[key] = node
            heappush(node_heap, node)

        n = 0
        while len(node_heap) > 1:
            pop1 = heappop(node_heap)
            pop2 = heappop(node_heap)

            node = Node(n+1, pop1.num + pop2.num)
            pop1.parent = (node, 'l')
            pop2.parent = (node, 'r')
            heappush(node_heap, node)
            n += 1

        word_parents = [None] * len(id_to_token)
        signals = [None] * len(id_to_token)
        for i, node in enumerate(word_nodes):
            leaf = node
            parents = []
            signal = []
            while leaf.parent != None:
                signal.append(1 if leaf.parent[1] == 'l' else -1)
                parents.append(leaf.parent[0].id-1)
                leaf = leaf.parent[0]

            word_parents[i] = parents[::-1]
            signals[i] = signal[::-1]

        self.model_save['huffle_tree'] = (n, word_parents, signals)





