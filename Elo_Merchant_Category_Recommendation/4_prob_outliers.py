# -*- coding: utf-8 -*-
"""
@author: xp
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

n_folds = 11
kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
#kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=42)

train_df = pd.read_csv('G:/1/train_cle.csv')
test_df = pd.read_csv('G:/1/test_cle.csv')
target = train_df['outliers']

clfs = list()
score = 0
oof_preds = np.zeros(train_df.shape[0])

FEATS_EXCLUDED = [
    'first_active_month', 'target', 'card_id', 'outliers',
    'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
    'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
    'OOF_PRED', 'month_0']      

features = [c for c in train_df.columns if c not in FEATS_EXCLUDED]
best_params = {'boosting_type': 'dart',
               'cat_l2': 7.232497734779235,
               'cat_smooth': 16.659594663169624,
               'colsample_bytree': 0.5179190185261051,
               'learning_rate': 0.48029619861737327,
               'max_cat_threshold': 11,
               'max_cat_to_onehot': 4,
               'max_depth': 70,
               'metric_freq': 11,
               'min_child_samples': 187,
               'min_child_weight': 27.410240796071633,
               'min_data_per_group': 74,
               'min_split_gain': 0.0332800062644832,
               'num_leaves': 87,
               'objective': 'regression_l2',
               'reg_alpha': 12.606656968904204,
               'reg_lambda': 12.351358114497039,
               'subsample': 0.9702576262397172,
               'subsample_for_bin': 180000,
               'subsample_freq': 37}
   
best_params.update({'n_estimators': 20000})
      

for num, (trn_idx, val_idx) in enumerate(kfolds.split(train_df.values, target.values)):
    print('no {} of {} folds'.format(num, n_folds))
       
    X_train, y_train = train_df.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = train_df.iloc[val_idx][features], target.iloc[val_idx]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(
            X_train, y_train,
            # eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_set=[(X_valid, y_valid)],
            verbose=50, eval_metric='rmse',
            early_stopping_rounds=500
        )

    clfs.append(model)
    oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
    score = mean_squared_error(target.values, oof_preds) ** 0.5
    print("score: ", score)

del X_train, y_train, X_valid, y_valid
gc.collect()


def predict_cross_validation(test, clfs, ntree_limit=None):
    sub_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):

        num_tree = 10000
        if not ntree_limit:
            ntree_limit = num_tree

        if isinstance(model, lgb.sklearn.LGBMRegressor):
            if model.best_iteration_:
                num_tree = min(ntree_limit, model.best_iteration_)

            test_preds = model.predict(test, raw_score=True, num_iteration=num_tree)


        sub_preds += test_preds

    sub_preds = sub_preds / len(clfs)
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret


file_template = '{score:.6f}_{model_key}_cv{fold}_{timestamp}'
file_stem = file_template.format(
            score=score,
            model_key='LGBM',
            fold=n_folds,
            timestamp=datetime.now().strftime('%Y-%m-%d-%H-%M')
            )

filename = 'subm_{}.csv'.format(file_stem)
print('save to {}'.format(filename))
subm = predict_cross_validation(test_df[features], clfs)
subm = subm.to_frame('target')
subm.to_csv(filename, index=False)

