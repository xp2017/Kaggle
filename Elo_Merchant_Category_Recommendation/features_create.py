# -*- coding: utf-8 -*-
"""
@author: xp
"""
import numpy as np
import pandas as pd
from datetime import datetime 

DATE_TODAY = datetime(2019, 1, 26)
def process_date(df):
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['hour'] = df['purchase_date'].dt.hour
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['weekday'] = df['purchase_date'].dt.weekday
    df['weekend'] = (df['purchase_date'].dt.weekday >= 5).astype(int)
    return df


def dist_holiday(df, col_name, date_holiday, date_ref, period=100):
    df[col_name] = np.maximum(np.minimum((pd.to_datetime(date_holiday) - df[date_ref]).dt.days, period), 0)
    

def historical_transactions(df):
    """
    preprocessing historical transactions
    """
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
    ]

    # agg
    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['max', 'min', 'mean', 'var', 'skew'],
        'authorized_flag': ['mean'],
        'weekend': ['mean'], # overwrite
        'weekday': ['mean'], # overwrite
        'day': ['nunique', 'mean', 'min'], # overwrite
        'category_1': ['mean'],
        'category_2': ['mean'],
        'category_3': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['sum', 'mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Mothers_Day_2017': ['mean', 'sum'],
        'fathers_day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Valentine_Day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
    }
    aggs.update(aggs_specific)

    
    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)


    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(np.int16)
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(np.int16)
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C':2}).astype(np.int16)

    df['price'] = df['purchase_amount'] / df['installments']

    df = process_date(df)

    for d_name, d_day in holidays:
        dist_holiday(df, d_name, d_day, 'purchase_date')

    df['month_diff'] = (DATE_TODAY - df['purchase_date']).dt.days // 30
    df['month_diff'] += df['month_lag']

    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    for col in ['category_2', 'category_3']:
        df[col + '_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col + '_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col + '_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col + '_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']
    
    df = df.reset_index().groupby('card_id').agg(aggs)
    
    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])
    df.columns = ['hist_' + c for c in df.columns]

    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']

    df['hist_purchase_date_diff'] = (df['hist_purchase_date_max'] - df['hist_purchase_date_min']).dt.days
    df['hist_purchase_date_average'] = df['hist_purchase_date_diff'] / df['hist_card_id_size']
    df['hist_purchase_date_uptonow'] = (DATE_TODAY - df['hist_purchase_date_max']).dt.days
    df['hist_purchase_date_uptomin'] = (DATE_TODAY - df['hist_purchase_date_min']).dt.days

    return df



def new_merchant_transactions(df):
 
    na_dict = {
        'category_2': 1.,
        'category_3': 'A',
        'merchant_id': 'M_ID_00a6ca8a8a',
    }

    holidays = [
        ('Christmas_Day_2017', '2017-12-25'),  # Christmas: December 25 2017
        # ('Mothers_Day_2017', '2017-06-04'),  # Mothers Day: May 14 2017
        # ('fathers_day_2017', '2017-08-13'),  # fathers day: August 13 2017
        ('Children_day_2017', '2017-10-12'),  # Childrens day: October 12 2017
        # ('Valentine_Day_2017', '2017-06-12'),  # Valentine's Day : 12th June, 2017
        ('Black_Friday_2017', '2017-11-24'),  # Black Friday: 24th November 2017
        ('Mothers_Day_2018', '2018-05-13'),
    ]
    
    aggs = dict()
    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    aggs.update({col: ['nunique'] for col in col_unique})

    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']
    aggs.update({col: ['nunique', 'mean', 'min', 'max'] for col in col_seas})

    aggs_specific = {
        'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
        'installments': ['sum', 'max', 'mean', 'var', 'skew'],
        'purchase_date': ['max', 'min'],
        'month_lag': ['max', 'min', 'mean', 'var', 'skew'],
        'month_diff': ['mean', 'var', 'skew'],
        'weekend': ['mean'],
        'month': ['mean', 'min', 'max'],
        'weekday': ['mean', 'min', 'max'],
        'category_1': ['mean'],
        'category_2': ['mean'],
        'category_3': ['mean'],
        'card_id': ['size', 'count'],
        'price': ['mean', 'max', 'min', 'var'],
        'Christmas_Day_2017': ['mean', 'sum'],
        'Children_day_2017': ['mean', 'sum'],
        'Black_Friday_2017': ['mean', 'sum'],
        'Mothers_Day_2018': ['mean', 'sum'],
        'duration': ['mean', 'min', 'max', 'var', 'skew'],
        'amount_month_ratio': ['mean', 'min', 'max', 'var', 'skew'],
    }
    aggs.update(aggs_specific)

    df.fillna(na_dict, inplace=True)
    df['installments'].replace({
        -1: np.nan, 999: np.nan}, inplace=True)

    df['purchase_amount'] = df['purchase_amount'].apply(lambda x: min(x, 0.8))

    df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(int).astype(np.int16)
    df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int).astype(np.int16)

    df['price'] = df['purchase_amount'] / df['installments']

    df = process_date(df)
    for d_name, d_day in holidays:
        dist_holiday(df, d_name, d_day, 'purchase_date')

    df['month_diff'] = (DATE_TODAY - df['purchase_date']).dt.days // 30
    df['month_diff'] += df['month_lag']

    df['duration'] = df['purchase_amount'] * df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount'] / df['month_diff']

    for col in ['category_2', 'category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
        df[col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    df = df.reset_index().groupby('card_id').agg(aggs)

    df.columns = pd.Index([e[0] + "_" + e[1] for e in df.columns.tolist()])
    df.columns = ['new_' + c for c in df.columns]

    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    
    df['new_purchase_date_diff'] = (df['new_purchase_date_max'] - df['new_purchase_date_min']).dt.days
    df['new_purchase_date_average'] = df['new_purchase_date_diff'] / df['new_card_id_size']
    df['new_purchase_date_uptonow'] = (DATE_TODAY - df['new_purchase_date_max']).dt.days
    df['new_purchase_date_uptomin'] = (DATE_TODAY - df['new_purchase_date_min']).dt.days

    return df
 

def additional_features(df):
    
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features = [
        'hist_purchase_date_max', 'hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']
    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    
    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']

    df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']

    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']

    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    
    df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
        
    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
    
    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
    
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']
    df['CLV_sq'] = df['new_CLV'] * df['hist_CLV']

    return df       