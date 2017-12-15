#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier,LinearRegression,RidgeCV,LassoCV
from sklearn.svm import LinearSVC  
import itertools

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


def calScore(pre,label):
	if len(pre)==len(label):
		score=1.0*np.sum(pre==label)/len(label)
		#print score
		
if __name__=="__main__":
	def f(x):
		try:
			float(x)
			return 1
		except:
			return 0
	train=pd.read_csv("train.csv")
	test=pd.read_csv("test.csv")
	label=train['Survived'].ravel()
	del train['Survived']
	all_data=pd.concat((train,test))
	#all_data['Age']=StandardScaler().fit_transform(all_data['Age'][:,np.newaxis])
	temp=all_data['Name'].apply(lambda x:'women' if 'Mrs' in x  or 'Miss' in x else ('Mr' if 'Mr' in x else 'nullsex'))
	all_data['Name']=all_data['Name'].apply(lambda x:'Mrs' if 'Mrs' in x else ('Mr' if 'Mr' in x else ('Miss' if 'Miss' in x else 'nullsex')))
	all_data["Age"] = all_data.groupby("Name")["Age"].transform(lambda x: x.fillna(x.mean()))
	all_data['Name']=temp
	all_data['Fare'].fillna(all_data['Fare'].mean(),inplace=True)
	#all_data['Fare']=StandardScaler().fit_transform(all_data['Fare'][:,np.newaxis])
	all_data['Cabin']=all_data['Cabin'].apply(lambda x:'nan' if x ==np.nan else str(x)[0])
	all_data['Embarked'].fillna(all_data['Embarked'].mode()[0],inplace=True)
	all_data['Ticket']=all_data['Ticket'].apply(lambda x:f(x))
	all_data['familysize']=all_data['SibSp']+all_data['Parch']
	all_data['Pclass']=all_data['Pclass'].astype(str)
	all_data=pd.get_dummies(all_data)

	train=all_data[all_data['PassengerId']<=891]
	test=all_data[all_data['PassengerId']>891]
	
	
	
	del train['PassengerId']
	testid=test['PassengerId']
	del test['PassengerId']
	ntrain=train.shape[0]
	ntest=test.shape[0]
	k=5
	SEED=0
	kf=KFold(ntrain,n_folds=k,random_state=SEED)
	
	class SklearnHelper(object):
		def __init__(self, clf, seed=0, params=None):
			params['random_state'] = seed
			self.clf = clf(**params)

		def train(self, x_train, y_train):
			self.clf.fit(x_train, y_train)

		def predict(self, x):
			return self.clf.predict(x)
		
		def fit(self,x,y):
			return self.clf.fit(x,y)
		
		def feature_importances(self,x,y):
			print(self.clf.fit(x,y).feature_importances_)
	
	
	#针对一个模型进行处理，输出处理后的training set和test set
	def get_oof(clf,x_train,y_train,x_test):
		#用于保存最后k折之后的训练集
		oof_train=np.zeros((ntrain,))
		#用于保存测试集预测之后的平均
		oof_test=np.zeros((ntest,))
		oof_test_skf=np.empty((k,ntest))
		for i,(train_index,test_index) in enumerate(kf):
			#将原有的训练集切分为训练集和测试集
			x_tr=x_train[train_index]
			y_tr=y_train[train_index]
			x_te=x_train[test_index]
			clf.fit(x_tr,y_tr)
			oof_train[test_index]=clf.predict(x_te)
			oof_test_skf[i,:]=clf.predict(x_test)	#固定行填充，循环一次，填充一次
		oof_test[:]=oof_test_skf.mean(axis=0)
		#由一行转置为变为一列
		return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
		
	# Random Forest parameters
	rf_params = {
		'n_jobs': -1,
		'n_estimators': 500,
		 'warm_start': True, 
		 #'max_features': 0.2,
		'max_depth': 6,
		'min_samples_leaf': 2,
		'max_features' : 'sqrt',	
		'verbose': 0
	}
		
	# Extra Trees Parameters
	et_params = {
		'n_jobs': -1,
		'n_estimators':500,
		#'max_features': 0.5,
		'max_depth': 8,
		'min_samples_leaf': 2,
		'verbose': 0
	}
	
	# AdaBoost parameters
	ada_params = {
		'n_estimators': 500,
		'learning_rate' : 0.75
	}
	
	# Gradient Boosting parameters
	gb_params = {
		'n_estimators': 500,
		 #'max_features': 0.2,
		'max_depth': 5,
		'min_samples_leaf': 2,
		'verbose': 0
	}
	
	# Support Vector Classifier parameters 
	svc_params = {
		'kernel' : 'linear',
		'C' : 0.025
    }
	x_train=train.values
	y_train=label
	x_test=test.values
	rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
	et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
	ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
	gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
	svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
	
	et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
	rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
	ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
	gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
	svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
	
	
	base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
	
	
	x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
	x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
	
	
	gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
	n_estimators= 2000,
	max_depth= 4,
	min_child_weight= 2,
	#gamma=1,
	gamma=0.9,                        
	subsample=0.8,
	colsample_bytree=0.8,
	objective= 'binary:logistic',
	nthread= -1,
	scale_pos_weight=1).fit(x_train, y_train)
	predictions = gbm.predict(x_test)
	
	
	StackingSubmission = pd.DataFrame({ 'PassengerId': testid,
                            'Survived': predictions })
	StackingSubmission.to_csv("StackingSubmission.csv", index=False)
	
