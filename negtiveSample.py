#coding=utf-8
from random import *
import pandas as pd
items_pool=dict()
class LatentFactorModel:
    users_array=
    items_array=
    user_items_map=dict()
    items_pool_map=dict()
    P_map=dict()
    Q_map=dict()
    def initModel(self, user_items_map, F_int):
        '''
        description:  用参数 user_items 初始化P 和 Q
        parameters: user_items: 每个用户的交互物品集合;F: 分类的个数 
        '''
        for user_int in range(len(users_array)):
            temp_map=dict()
            for f_int in range(F_int):
                temp_map[f_int]=0
            P_map[user_int]=temp_map
        for f_int in range(F_int):
            temp_map=dict()
            for items_int in range(len(items_array)):
                temp_map[items_int]=0
            Q[f_int]=temp_map
        return
    def __init__(self):
        return

    def randomSelectNegativeSamples(self, items):
        '''
        description: 从 item_pool 中采样出与 items 数量相近的负样本，并与正样本一起放入到集合中
        parameters: items: 维护某一个用户与物品的交互集合; items_pool: 维护一个物品集合，这个集合中同一物品出现的次数跟他的流行度成正比
        return: 包含 items 在内的正负样本集合
        '''
        ret = dict()
        for i in items.keys():
            ret[i] = 1
        n=0
        for i in range(0, len(items) * 3):
            item = items_pool[random.randint(0, len(items_pool) - 1)]
            if item in ret:
                continue
            ret[item] = 0
            n+=1
            if n > len(items):
                break
        return ret

    def recommend(self, user):
        '''
        description:  给定用户 user，计算每个物品 i 对该用户的推荐度
        parameters: user：用户的 id；P：一个 map（dict） 的数组，这个 map 的 key 为 f，value 为 puf；Q：一个 map（dict） 数组，key 为 f，value 为 qfi
        '''
        rank = dict()
        for f, puf in self.P[user].items():
            for i, qfi in self.Q[f].items():
                if i not in rank:
                    rank[i] += puf * qfi
        return rank

    def latentFactorModel(self, user_items, F, N, alpha, lmd):
        '''
        description: 用梯度下降的方法得到 P 和 Q 
        parameters: user_items: 一个用户和与这个用户交互过的所有物品集合; F: 要将 item 划分成多少个类; N: 迭代的次数; alpha:每次迭代更新的步长; lmd:正则化系数;
        '''
        [self.P, self.Q] = self.initModel(user_items, F)
        for step in range(0,N):
            for user, items in user_items.items():
                samples = self.randomSelectNegativeSamples(items)
                for item, rui in samples.items():
                    eui = rui - Predict(user, item)
                    for f in range(0, F):
                        P[user][f] += alpha * (eui * Q[item][f] - lmd * P[user][f])
                        Q[item][f] += alpha * (eui * P[user][f] - lmd * Q[item][f])
        alpha *= 0.9
    if __name__=='__main__':
        print('hello')
