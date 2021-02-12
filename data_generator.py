# -*- coding: utf-8 -*-
"""
@author: Yanagisawa
"""
import sys
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

class DataGenerator:
    def __init__(self, random_seed, time_steps, train_start_index, len_train, len_test, index_all, len_val = 10, is_val = False):
        '''
            データ作成に関わるクラス        

        Parameters
        ----------
        random_seed : TYPE
            乱数シード.
        time_steps : TYPE
            過去データの参照期間.
        train_start_index:
            開始時点のインデックス
        train_start_date:
            訓練期間の開始時点
        len_train : datetime
            訓練期間数.
        len_test : datetime
            テスト期間数.
        index_all : DatetimeIndex
            全データの日付            
        len_val : datetime
            ヴァリデーション期間数.
        is_val : bool
            ヴァリデーションのデータセット作成を行うかどうか
        '''
        if type(index_all) is not pd.core.indexes.datetimes.DatetimeIndex:
            sys.exit('データフレームのインデックスに対し，pd.to_datetimeを実行してください')
        if train_start_index-time_steps < 0:
            sys.exit('train_start_indexはtime_stepsより大きい値を指定してください')
        
        self.random_seed = random_seed
        self.time_steps = time_steps
        self.train_start_index = train_start_index
        self.len_train = len_train
        self.len_test = len_test
        self.len_val = len_val
        self.is_val = is_val
        
        # 訓練データとテストデータの日付作成
        self.train_start_date            = index_all[train_start_index:,][0]
        self.train_start_date_timeseries = index_all[train_start_index-time_steps:,][0]        
        self.train_end_date              = index_all[:(train_start_index+len_train),][-1]
        
        if is_val:               
            self.val_start_date             = index_all[(train_start_index+len_train):,][0]
            self.val_start_date_timeseries  = index_all[(train_start_index+len_train-time_steps):,][0]
            self.val_end_date               = index_all[:(train_start_index+len_train+len_val),][-1]
            self.test_start_date            = index_all[(train_start_index+len_train+len_val):,][0]            
            self.test_start_date_timeseries = index_all[(train_start_index+len_train+len_val-time_steps):,][0]
            self.test_end_date              = index_all[:(train_start_index+len_train+len_val+len_test),][-1]            
        else:                
            self.test_start_date            = index_all[(train_start_index+len_train):,][0]            
            self.test_start_date_timeseries = index_all[(train_start_index+len_train-time_steps):,][0]
            self.test_end_date              = index_all[:(train_start_index+len_train+len_test),][-1]            
    
    def output_split_status(self, is_val, col_name):
        if is_val:
            output = pd.DataFrame([self.train_start_date.date(), self.train_end_date.date(),
                                   self.val_start_date.date(), self.val_end_date.date(),
                                   self.test_start_date.date(), self.test_end_date.date()],
                                   index = ['訓練開始時点_ハイパラ', '訓練終了時点_ハイパラ',
                                            '評価開始時点_ハイパラ', '評価終了時点_ハイパラ',
                                            'テスト開始時点_ハイパラ', 'テスト終了時点_ハイパラ'], columns = [col_name])        
        else:
            output = pd.DataFrame([self.train_start_date.date(), self.train_end_date.date(),
                                   self.test_start_date.date(), self.test_end_date.date()],
                                   index = ['訓練開始時点_再学習', '訓練終了時点_再学習',
                                            'テスト開始時点_再学習', 'テスト終了時点_再学習'], columns = [col_name])
        return output
    
    def visualize_split_status(self):
        if self.is_val:
            print(f'訓練期間{self.train_start_date.date()}<-->{self.train_end_date.date()}',
                  f'\n評価期間{self.val_start_date.date()}<-->{self.val_end_date.date()}',
                  f'\nテスト期間{self.test_start_date.date()}<-->{self.test_end_date.date()}')
        else:
            print(f'訓練期間{self.train_start_date.date()}<-->{self.train_end_date.date()}',
                  f'\nテスト期間{self.test_start_date.date()}<-->{self.test_end_date.date()}')
            
    def split_train_test(self, df, is_time_steps = True):
        '''
            訓練データとテストデータに分割する

        Parameters
        ----------
        df : data.frame
        is_time_steps : RNN系モデルの特徴量として使用するデータの場合はTrue
        Returns
        -------
        df_train : data.frame
            訓練データ
        df_test : data.frame
            テストデータ

        '''

        if self.is_val:
            if is_time_steps:
                df_train, df_val, df_test = \
                    df[self.train_start_date_timeseries:self.train_end_date], \
                    df[self.val_start_date_timeseries:self.val_end_date], \
                    df[self.test_start_date_timeseries:self.test_end_date]
            else:
                df_train, df_val, df_test = \
                    df[self.train_start_date:self.train_end_date], \
                    df[self.val_start_date:self.val_end_date], \
                    df[self.test_start_date:self.test_end_date]
            
            return df_train, df_val, df_test        
        
        else:            
            if is_time_steps:
                df_train, df_test = \
                    df[self.train_start_date_timeseries:self.train_end_date], \
                    df[self.test_start_date_timeseries:self.test_end_date]
            else:
                df_train, df_test = \
                    df[self.train_start_date:self.train_end_date], \
                    df[self.test_start_date:self.test_end_date]

            return df_train, df_test
    
    def scale(self, df_train, df_test, df_val=[], method='StandardScaler'):
        '''
            MinMax変換により基準化を行う関数

        Parameters
        ----------
        df_train : TYPE
            訓練データ
        df_test : TYPE
            テストデータ
        method : 
            基準化方法を指定

        Returns
        -------
        df_train_scaled : TYPE
            基準化後の訓練データ
        df_test_scaled : TYPE
            基準化後のテストデータ

        '''
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()      
            
        scaler.fit(df_train)        
        df_train_scaled = pd.DataFrame(scaler.transform(df_train), index = df_train.index, columns = df_train.columns)
        df_test_scaled = pd.DataFrame(scaler.transform(df_test), index = df_test.index, columns = df_test.columns)
        if len(df_val)>0:
            df_val_scaled = pd.DataFrame(scaler.transform(df_val), index = df_val.index, columns = df_val.columns)            
            return df_train_scaled, df_val_scaled, df_test_scaled 
        else:
            return df_train_scaled, df_test_scaled 
    
    def build_timeseries(self, df, target_names):
        '''
            torch型の特徴量データを作成
            (時点，過去の参照期間，特徴量, 目的変数の数)
        '''
        dim_0 = df.shape[0] - self.time_steps
        dim_1 = df.shape[1]
        dim_2 = len(target_names)
        X = np.zeros((dim_0, self.time_steps, dim_1, dim_2))
        
        for j in range(dim_2):        
            for i in range(dim_0):
                X[i,:,:,j] = df[i:self.time_steps+i]
               
        output = torch.from_numpy(X).type(torch.Tensor)
        
        return output

    def make_batch(self, X, Y, batch_size):
        '''
            バッチデータ作成

        Parameters
        ----------
        X, Y : data.frame
        batch_size : TYPE
            バッチサイズ.

        Returns
        -------
        loader : TYPE
            DESCRIPTION.
        loader_fit : TYPE
            DESCRIPTION.

        '''
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        
        dataset = torch.utils.data.TensorDataset(X, Y)            
        loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size,
                                             shuffle = False, worker_init_fn = self.random_seed)        
        loader_fit = torch.utils.data.DataLoader(dataset = dataset, batch_size = X.shape[0], 
                                                 shuffle = False, worker_init_fn = self.random_seed)
        
        return loader, loader_fit
    
    def make_dataloader(self, dg, df, batch_size, target_names, is_target_scale=True):
        dg.visualize_split_status()
        df_train_, df_test_ = dg.split_train_test(df, is_time_steps = True)
        df_train_scaled, df_test_scaled = dg.scale(df_train_, df_test_)
        
        df_train = dg.build_timeseries(df_train_scaled, target_names)
        df_test = dg.build_timeseries(df_test_scaled, target_names)

        Y_train_, Y_test_ = dg.split_train_test(df[target_names], is_time_steps = False)
        if is_target_scale:
            Y_train, Y_test = dg.scale(Y_train_, Y_test_)
        else:
            Y_train, Y_test = Y_train_.copy(), Y_test_.copy()
            
        Y_train = torch.from_numpy(Y_train.values.astype(np.float32)).clone()
        Y_test = torch.from_numpy(Y_test.values.astype(np.float32)).clone()
                           
        loader_train, loader_train_fit = dg.make_batch(df_train, Y_train, batch_size)
        loader_test, loader_test_fit = dg.make_batch(df_test, Y_test, batch_size)
        
        return loader_train, loader_train_fit, loader_test, loader_test_fit
