# -*- coding: utf-8 -*-
"""
@author: xp
"""
import gc
import pandas as pd
from datetime import datetime 
from reduce_memory import  reduce_mem_usage
from one_hot_encoder import one_hot_encoder
from feature_create import historical_transactions, new_merchant_transactions, additional_features


train_df = pd.read_csv('G:/1/inputs/train.csv', index_col=['card_id'], parse_dates=['first_active_month'])
test_df = pd.read_csv('G:/1/inputs/test.csv', index_col=['card_id'], parse_dates=['first_active_month'])

train_df['outliers'] = 0
train_df.loc[train_df['target'] < -30, 'outliers'] = 1


DATE_TODAY = datetime(2019, 1, 26)
for df in [train_df, test_df]:
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (DATE_TODAY - df['first_active_month']).dt.days

    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    for f in feature_cols:    
        df['days_' + f] = df['elapsed_time'] * df[f]
        df['days_' + f + '_ratio'] = df[f] / df['elapsed_time']

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    df_feats = df.reindex(columns=feature_cols)
    df['features_sum'] = df_feats.sum(axis=1)
    df['features_mean'] = df_feats.mean(axis=1)
    df['features_max'] = df_feats.max(axis=1)
    df['features_min'] = df_feats.min(axis=1)
    df['features_var'] = df_feats.std(axis=1)
    df['features_prod'] = df_feats.product(axis=1)
    df = reduce_mem_usage(df)
    
    
feature_cols = ['feature_1', 'feature_2', 'feature_3']
for f in feature_cols:
    order_label = train_df.groupby([f])['outliers'].mean()
    train_df[f] = train_df[f].map(order_label)
    test_df[f] = test_df[f].map(order_label) 



history_df= pd.read_csv('G:/1/inputs/historical_transactions.csv')
history_df = historical_transactions(history_df)

history_df = reduce_mem_usage(history_df)
new_df = pd.read_csv('G:/1/inputs/new_merchant_transactions.csv') 
new_df = new_merchant_transactions(new_df) 
new_df = reduce_mem_usage(new_df)
transaction_df = pd.concat([new_df, history_df], axis=1)
del new_df, history_df
gc.collect()

train_df = train_df.join(transaction_df, how='left', on='card_id')
test_df = test_df.join(transaction_df, how='left', on='card_id')
del transaction_df
gc.collect()


train_df = additional_features(train_df)
train_df = reduce_mem_usage(train_df)
train_df.to_csv("G:/1/train_cle.csv", index=False)
del train_df
gc.collect()
test_df = additional_features(test_df)
test_df = reduce_mem_usage(test_df)

test_df.to_csv("G:/1/test_cle.csv", index=False)
del test_df
gc.collect()