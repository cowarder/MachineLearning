#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
data_path = '../input/'

print('Loading data...')
train=pd.read_csv(data_path+'train.csv')
test=pd.read_csv(data_path+'test.csv')
songs=pd.read_csv(data_path+'songs.csv')
members=pd.read_csv(data_path+'members.csv',parse_dates=['registration_init_time','expiration_date'])
song_extra=pd.read_csv(data_path+'song_extra_info.csv')
print("Done loading data")


#分析msno
def split_user_id(x):
	return str(x)[0]
train['split_user_id']=train['msno'].apply(split_user_id)
test['split_user_id']=test['msno'].apply(split_user_id)

print('Merging data...')
def isrc_to_year(x):
	if type(x)==str:
		if int(str(x)[5:7])>17:
			return 2000+int(str(x)[5:7])
		else:
			return 1990+int(str(x)[5:7])
	else:
		return np.nan

"""
#歌曲回听比例
song_count=train[['song_id','target']].groupby(['song_id']).agg(['sum','count'])
song_count['song_sum']=song_count['target','sum']
song_count['song_count']=song_count['target','count']
song_count.drop(['target'],axis=1,inplace=True)
song_count['song_percent']=100.0*song_count['song_sum']/song_count['song_count']
song_count['song_percent']=song_count['song_percent'].replace(100.0,np.nan)
print(song_count['song_id'])
train=train.merge(song_count,left_on='song_id',right_index=True,how='left')
test=test.merge(song_count,left_on='song_id',right_index=True,how='left')
print(train['song_percent'])
"""

"""
#用户回听比例
user_count=train[['msno','target']].groupby(['msno']).agg(['sum','count'])
user_count['user_sum']=user_count['target','sum']
user_count['user_count']=user_count['target','count']
user_count.drop(['target'],axis=1,inplace=True)
user_count['user_percent']=100.0*user_count['user_sum']/user_count['user_count']
user_count['user_percent']=user_count['user_percent'].replace(100.0,np.nan)
train=train.merge(user_count,left_on='msno',right_index=True,how='left')
test=test.merge(user_count,left_on='msno',right_index=True,how='left')
"""

def isrc_to_country(x):
	return str(x)[:2]
	
song_extra['country']=song_extra['isrc'].apply(isrc_to_country)
song_extra['song_year']=song_extra['isrc'].apply(isrc_to_year)
song_extra.drop(['isrc'],axis=1,inplace=True)

members['reg_year']=members['registration_init_time'].dt.year
members['reg_month']=members['registration_init_time'].dt.month
members['reg_day']=members['registration_init_time'].dt.day
members['exp_year']=members['expiration_date'].dt.year
members['exp_month']=members['expiration_date'].dt.month
members['exp_day']=members['expiration_date'].dt.day
members['span_days']=(members['expiration_date']-members['registration_init_time']).dt.days
members.drop(['registration_init_time'], axis=1,inplace=True)
members.drop(['expiration_date'], axis=1,inplace=True)


#对缺失值做一些处理
train['source_screen_name'].fillna('nan',inplace=True)
train['source_system_tab'].fillna('nan',inplace=True)
train['source_type'].fillna('nan',inplace=True)
test['source_screen_name'].fillna('nan',inplace=True)
test['source_system_tab'].fillna('nan',inplace=True)
test['source_type'].fillna('nan',inplace=True)


#median_age=members[(members['bd']>5)&(members['bd']<100)]['bd'].median()
def norm_age(x):
	if x>100 or x<5:
		return 0
	else:
		return x
		
members['bd']=members['bd'].apply(norm_age)

train=train.merge(members,on='msno',how='left')
test=test.merge(members,on='msno',how='left')
train=train.merge(songs,on='song_id',how='left')
test=test.merge(songs,on='song_id',how='left')
train=train.merge(song_extra,on='song_id',how='left')
test=test.merge(song_extra,on='song_id',how='left')

#流派数量特征
def genre_id_count(x):
	if x=='no_genre':
		return 0
	else:
		return x.count('|') + 1

train['genre_ids'].fillna('no_genre',inplace=True)
test['genre_ids'].fillna('no_genre',inplace=True)
train['genre_count']=train['genre_ids'].apply(genre_id_count)
test['genre_count']=test['genre_ids'].apply(genre_id_count)

def split_genre(x):
	return str(x).split('|')[0]
train['genre_ids']=train['genre_ids'].apply(split_genre)
test['genre_ids']=test['genre_ids'].apply(split_genre)

#歌手数量特征
def artist_count(x):
	if x=='no_artist':
		return 0
	else:
		return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
train['artist_name'].fillna('no_artist',inplace=True)
test['artist_name'].fillna('no_artist',inplace=True)
train['artist_count']=train['artist_name'].apply(artist_count)
test['artist_count']=test['artist_name'].apply(artist_count)


#曲作者数量特征
def composer_count(x):
	if x=='no_composer':
		return 0
	else:
		return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)
train['composer']=train['composer'].apply(composer_count)
test['composer']=test['composer'].apply(composer_count)


#词作者数量特征
def lyricist_count(x):
	if x=='no_lyricist':
		return 0
	else:
		return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricist']=train['lyricist'].apply(lyricist_count)
test['lyricist']=test['lyricist'].apply(lyricist_count)


#歌手是否是featureed特征和歌曲是否是featureed特征
def is_featured(x):
	if 'feat' in str(x):
		return 1
	else:
		return 0

train['artist_featured'] = train['artist_name'].apply(is_featured)
test['artist_featured'] = test['artist_name'].apply(is_featured)
train['song_featured']=train['name'].apply(is_featured)
test['song_featured']=test['name'].apply(is_featured)
"""
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
def is_chinese(x):
	if zhPattern.findall(x):
		return 1
	else:
		return 0
		
train['is_chinese']=train['name'].apply(is_chinese)
test['is_chinese']=test['name'].apply(is_chinese)

"""
train.drop('name',axis=1,inplace=True)
test.drop('name',axis=1,inplace=True)

"""
#名字中切分符号数量特征
def count_add_split_token(x):
	return x.count('\\')+x.count('+')
train['song_token']=train['song_id'].apply(count_add_split_token)
test['song_token']=test['song_id'].apply(count_add_split_token)
train['user_token']=train['msno'].apply(count_add_split_token)
test['user_token']=test['msno'].apply(count_add_split_token)
"""

#歌曲播放量特征
dict_count_song_train={k:v for k,v in train['song_id'].value_counts().iteritems()}
dict_count_song_test={k:v for k,v in test['song_id'].value_counts().iteritems()}

def count_songs(x):
	try:
		return dict_count_song_train[x]+dict_count_song_test[x]
	except KeyError:
		try:
			return dict_count_song_train[x]
		except KeyError:
			try:
				return dict_count_song_test[x]
			except:
				return 0
train['count_songs']=train['song_id'].apply(count_songs)
test['count_songs']=test['song_id'].apply(count_songs)

#歌手歌曲量特征
dict_count_artist_train={k:v for k,v in train['artist_name'].value_counts().iteritems()}
dict_count_artist_test={k:v for k,v in test['artist_name'].value_counts().iteritems()}


def count_artist(x):
	try:
		return dict_count_artist_train[x]+dict_count_artist_test[x]
	except KeyError:
		try:
			return dict_count_artist_train[x]
		except KeyError:
			try:
				return dict_count_artist_test[x]
			except:	
				return 0
train['count_artist']=train['artist_name'].apply(count_artist)
test['count_artist']=test['artist_name'].apply(count_artist)


#语言是17或者45
def song_lang_bool(x):
	if '17.0' in str(x) or '45.0' in str(x):
		return 1
	else:
		return 0
train['song_lang_boool']=train['language'].apply(song_lang_bool)
test['song_lang_boool']=test['language'].apply(song_lang_bool)

#是否长度大于平均长度
mean_song_length=np.mean(train['song_length'])
def smaller_mean(x):
	if x<mean_song_length:
		return 1
	else:
		return 0
train['song_length']=train['song_length'].apply(smaller_mean)
test['song_length']=test['song_length'].apply(smaller_mean)


train['city']=train['city'].apply(str)
train['registered_via']=train['registered_via'].apply(str)
train['language']=train['language'].apply(str)
test['city']=test['city'].apply(str)
test['registered_via']=test['registered_via'].apply(str)
test['language']=test['language'].apply(str)

for col in train.columns:
	if train[col].dtypes=='object':
		train[col]=train[col].astype('category')
for col in test.columns:
	if test[col].dtypes=='object':
		test[col]=test[col].astype('category')
print('Done merging data')

print('Start training...')
ids=test['id']
x_train=train.drop(['target'],axis=1)
y_train=train['target'].values
x_test=test.drop(['id'],axis=1)
final_data=lgb.Dataset(x_train,y_train)
watch_data=lgb.Dataset(x_train,y_train)

"""
x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train)

lgb_train = lgb.Dataset(x_tr, y_tr)
lgb_val = lgb.Dataset(x_val, y_val)
"""

params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 10,
        'num_rounds': 200,
        'metric' : 'auc'
    }
	
model = lgb.train(params, train_set=final_data,  valid_sets=watch_data, verbose_eval=5)
result=model.predict(x_test)
sub=pd.DataFrame()
sub['id']=ids
sub['target']=result
sub.to_csv('submission_lgbm_avg.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done')