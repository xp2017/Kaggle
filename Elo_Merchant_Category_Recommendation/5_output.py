# -*- coding: utf-8 -*-
"""
@author: xp
"""

import pandas as pd

card_id = pd.read_csv('G:/1/inputs/test.csv', usecols=['card_id'])

model_with_outliers = pd.read_csv("G:/1/subm_3.648358_LGBM_cv11_2019-02-15-20-54.csv")
model_with_outliers['card_id'] = card_id['card_id'].values
#df_outlier_prob = pd.read_csv("G:/1/subm_0.099592_LGBM_cv11_2019-02-18-19-54.csv")
df_outlier_prob = pd.read_csv("G:/1/subm_0.100517_LGBM_cv11_2019-02-21-14-12.csv")
df_outlier_prob['card_id'] = card_id['card_id'].values

model_without_outliers = pd.read_csv("G:/1/subm_1.556036_LGBM_cv11_2019-02-18-19-18.csv")
model_without_outliers['card_id'] = card_id['card_id'].values

outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
most_likely_liers = model_with_outliers.merge(outlier_id,how='right')

for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\
    = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values
    
     
model_without_outliers.to_csv("G:/1/output/submission.csv", index=False)
#提交 3.688
#3.688
'''
head 300000 3.689
head 250000 2.688
head 200000 3.690
'''