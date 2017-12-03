#coding=utf-8
import random
import time
import pandas as pd
import numpy as np
import os

class LatentFactorModel:
    users_series = pd.Series()
    items_series = pd.Series()
    items_pool_series = pd.Series() # all movieIds which may repeat
    user_items_dict = dict() # every user andrelative set of items 
    user_negtive_samples_dict = dict()
    P_dict = dict()
    Q_dict = dict()
    F_int = 0
    N_int = 0
    alpha_float = 0
    lmd_float = 0
    dataset_path_string = '~/Datasets/ml-latest-small/'
    cost_file = open('./cost.file','w')
    
    def __init__(self, F_int, N_int, alpha_float, lmd_float):
        '''
        description:  初始化P_dict 和 Q_dict
        parameters:  F: 分类的个数; N: 迭代的次数; alpha:每次迭代更新的步长; lmd:正则化系数;
        '''
        self.F_int = F_int
        self.N_int = N_int
        self.alpha_float = alpha_float
        self.lmd_float = lmd_float

        ratings_dataframe = pd.read_csv(self.dataset_path_string+'ratings.csv')
        userId_series = ratings_dataframe['userId']
        movieId_series = ratings_dataframe['movieId']
        self.items_pool_series = movieId_series
        self.users_series = userId_series.unique()
        self.items_series = movieId_series.unique()

        for i_int in range(userId_series.size):
            userId_string = userId_series.iloc[i_int]
            movieId_string = movieId_series.iloc[i_int]
            if userId_string not in self.user_items_dict:
                self.user_items_dict[userId_string] = set()
            self.user_items_dict[userId_string].add(movieId_string)

        for userId_int in self.users_series:
            temp_dict = dict()
            for f_int in range(self.F_int):
                temp_dict[f_int] = random.random()
            self.P_dict[userId_int] = temp_dict
            # create the negtive samples
            user_items_set = self.user_items_dict[userId_int]
            negtive_samples_dict = self.randomSelectNegativeSamples(user_items_set)
            self.user_negtive_samples_dict[userId_int] = negtive_samples_dict
            
        for itemId_int in self.items_series:
            temp_dict = dict()
            for f_int in range(self.F_int):
                temp_dict[f_int] = random.random()
            self.Q_dict[itemId_int]=temp_dict
        return

    def __del__(self):
        self.cost_file.close()

    def print_out(self):
        '''
        description: just for test of class
        '''
        print("The users_series size: " + str(len(self.users_series)))
        print("The items_series size: " + str(len(self.items_series)))
        # print("The P_dict is: ")
        # print(self.P_dict)
        # print("The Q_dict is: ")
        # print(self.Q_dict)
        # Output the top-5 ralative movie for every class
        D_dict = dict()
        for f_int in range(self.F_int):
            D_dict[f_int] = dict()
        for item_int, items_dict in self.Q_dict.items():
            for f_int, qif_float in items_dict.items():
                D_dict[f_int][item_int] = qif_float

        outputSize_int = 5
        movies_dataframe = pd.read_csv('~/Datasets/ml-latest-small/movies.csv', index_col='movieId')

        for f_int in range(self.F_int):
            d_dict = D_dict[f_int]
            sorted_items_turple = sorted(d_dict.items(), key=lambda tuple:tuple[1], reverse=True)
            self.cost_file.write("Class " + str(f_int) + " :\n")
            for i_int in range(outputSize_int):
                item_int = sorted_items_turple[i_int][0]
                score_float = sorted_items_turple[i_int][1]
                row_series = movies_dataframe.loc[item_int]
                self.cost_file.write(str(item_int) + '\t' + row_series['title']+ "\t" + row_series['genres'] + "\t" + str(score_float) + "\n")
        return

    def randomSelectNegativeSamples(self, items_set):
        '''
        description: 从 item_pool 中采样出与 items 数量相近的负样本，并与正样本一起放入到集合中
        parameters: items: 维护某一个用户与物品的交互集合; items_pool: 维护一个物品集合，这个集合中同一物品出现的次数跟他的流行度成正比
        return: 包含 items 在内的正负样本集合
        '''
        ret_dict = dict()
        for i_int in items_set:
            ret_dict[i_int] = 1
        n_int = 0
        for i_int in range(len(items_set)):
            item_int = self.items_pool_series[random.randint(0, len(self.items_pool_series) - 1)]
            if item_int in ret_dict:
                continue
            ret_dict[item_int] = 0
            n_int += 1
            if n_int > len(items_set):
                break
        return ret_dict


    # def recommendScore(self, user_int):
    #     '''
    #     description:  给定用户 user，计算每个物品 i 对该用户的推荐度
    #     parameters: user：用户的 id；P：一个 map（dict） 的数组，这个 map 的 key 为 f，value 为 puf；Q：一个 map（dict） 数组，key 为 f，value 为 qfi
    #     '''
    #     rank_dict = dict()
    #     for f_int, puf_int in self.P_dict[user_int].items():
    #         for i_int, qfi_int in self.Q_dict[f_int].items():
    #             if i_int not in rank_dict:
    #                 rank_dict[i_int] += puf_int * qfi_int
    #     return rank_dict

    def costFunction(self):
        '''
        description: 根据P 和 Q 矩阵计算出所有 user 和对应的 negtive samples 的损失函数
        ''' 
        cost_float = 0
        for user_int in self.users_series:
            negtive_samples_dict = self.user_negtive_samples_dict[user_int]
            for item_int, rui_int in negtive_samples_dict.items():
                eui_float = rui_int - self.predict(user_int, item_int)
                cost_float += eui_float*eui_float
        for user_int in self.users_series:
            puf_dict = self.P_dict[user_int]
            for f_int in range(self.F_int):
                cost_float += self.lmd_float * puf_dict[f_int] * puf_dict[f_int] # the multiply of lambda can be done outside once
        for item_int in self.items_series:
            qif_dict = self.Q_dict[item_int]
            for f_int in range(self.F_int):
                cost_float += self.lmd_float * qif_dict[f_int] * qif_dict[f_int]
        return cost_float
 
    def predict(self, user_int, item_int):
        '''
        description: 根据 P 和 Q 矩阵计算出 user 对 item 的兴趣评分
        parameters: user_int: 指用户的 id, item_int: 指 item 的 id
        '''
        puf_dict = self.P_dict[user_int]
        qif_dict = self.Q_dict[item_int]
        ret_float = 0
        for f_int in range(self.F_int):
            ret_float += puf_dict[f_int]*qif_dict[f_int]
        return ret_float
        
    def trainModel(self):
        '''
        description: 用梯度下降的方法得到 P 和 Q
        parameters: user_items: 一个用户和与这个用户交互过的所有物品集合;
        '''
        for step_int in range(0, self.N_int):
            for user_int in self.users_series:
                negtive_samples_dict = self.user_negtive_samples_dict[user_int] 
                for item_int, rui_int in negtive_samples_dict.items():
                    eui_float = rui_int - self.predict(user_int, item_int)
                    for f_int in range(0, self.F_int):
                        deltaP_float = self.alpha_float * (eui_float * self.Q_dict[item_int][f_int] - self.lmd_float * self.P_dict[user_int][f_int])
                        deltaQ_float = self.alpha_float * (eui_float * self.P_dict[user_int][f_int] - self.lmd_float * self.Q_dict[item_int][f_int])
                        self.P_dict[user_int][f_int] += deltaP_float
                        self.Q_dict[item_int][f_int] += deltaQ_float
            cost_float = self.costFunction()
            self.cost_file.write(str(cost_float) + '\n');
        # alpha_float *= 0.9

if __name__=='__main__':
    lfm = LatentFactorModel(5,10,0.02,0.1)
    start = time.clock()
    lfm.trainModel()
    stop = time.clock()
    print(str(stop - start) + ' seconds')
    lfm.print_out()