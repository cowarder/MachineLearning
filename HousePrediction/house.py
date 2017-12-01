#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNet,TheilSenRegressor,HuberRegressor,RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import itertools
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
	
def calScore(pre,label):
	if len(pre)==len(label):
		score=np.sqrt(sum(pow(x-y,2) for x,y in zip(pre,label))/len(label))
		print score
	else:
		print "Their length are different"
def loadData():
	train=pd.read_csv('train.csv')
	test=pd.read_csv('test.csv')
	label=train['SalePrice']
	#id=train['Id']
	#train['Id']
	#print train.head(3)
	del train['SalePrice']
	train=pd.merge(train,test,how='outer')
	
	train=train.join(pd.get_dummies(train['MSSubClass'],prefix='mssubclass'))
	train['LotFrontage'].fillna(train['LotFrontage'].mean(),inplace=True)
	train['Alley'].fillna('nan',inplace=True)
	train['MasVnrType'].fillna('nan',inplace=True)
	train['MasVnrArea'].fillna(train['MasVnrArea'].mean(),inplace=True)
	train['BsmtQual'].fillna('nan',inplace=True)
	train['BsmtCond'].fillna('nan',inplace=True)
	train['BsmtExposure'].fillna('nan',inplace=True)
	train['BsmtFinType1'].fillna('nan',inplace=True)
	train['BsmtFinType2'].fillna('nan',inplace=True)
	train['Electrical'].fillna(sorted(train['Electrical'].value_counts(),reverse=True)[0],inplace=True)
	train['FireplaceQu'].fillna('nan',inplace=True)
	train['GarageType'].fillna('nan',inplace=True)
	train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(),inplace=True)
	train['GarageFinish'].fillna('nan',inplace=True)
	train['GarageQual'].fillna('nan',inplace=True)
	train['GarageCond'].fillna('nan',inplace=True)
	train['PoolQC'].fillna('nan',inplace=True)
	train['Fence'].fillna('nan',inplace=True)
	train['MiscFeature'].fillna('nan',inplace=True)
	train['BsmtFinSF1'].fillna(train['BsmtFinSF1'].mean(),inplace=True)
	train['BsmtFinSF2'].fillna(train['BsmtFinSF2'].mean(),inplace=True)
	train['BsmtUnfSF'].fillna(train['BsmtUnfSF'].mean(),inplace=True)
	train['TotalBsmtSF'].fillna(train['TotalBsmtSF'].mean(),inplace=True)
	train['BsmtFullBath'].fillna(1,inplace=True)
	train['BsmtHalfBath'].fillna(0,inplace=True)
	train['GarageCars'].fillna(2,inplace=True)
	train['GarageArea'].fillna(train['TotalBsmtSF'].mean(),inplace=True)
	#for col in train.columns:
	#	print col+':'+str(train[col].count())
	#对Alley进行分析，alley没有的居多，将其分为有和没有两个字段
	train['Alley']=train['Alley'].apply(lambda x:0 if type(x)==float else 1)
	train['LotFrontage']=(train['LotFrontage']-train['LotFrontage'].mean())/train['LotFrontage'].std()
	train['LotArea']=(train['LotArea']-train['LotArea'].mean())/train['LotArea'].std()
	train=train.join(pd.get_dummies(train['Alley'],prefix='alley'))
	train=train.join(pd.get_dummies(train['Condition2'],prefix='condition2'))
	train=train.join(pd.get_dummies(train['Utilities'],prefix='utilities'))
	train=train.join(pd.get_dummies(train['Street'],prefix='street'))
	train=train.join(pd.get_dummies(train['MSZoning'],prefix='mszoning'))
	train=train.join(pd.get_dummies(train['LotShape'],prefix='shape'))
	train=train.join(pd.get_dummies(train['LandContour'],prefix='contour'))
	train=train.join(pd.get_dummies(train['LotConfig'],prefix='config'))
	train=train.join(pd.get_dummies(train['LandSlope'],prefix='slope'))
	train=train.join(pd.get_dummies(train['Neighborhood'],prefix='neighbor'))
	train=train.join(pd.get_dummies(train['Condition1'],prefix='con1'))
	train=train.join(pd.get_dummies(train['BldgType'],prefix='dwtype'))
	train=train.join(pd.get_dummies(train['HouseStyle'],prefix='housestyle'))
	train=train.join(pd.get_dummies(train['OverallQual'],prefix='overqual'))
	train=train.join(pd.get_dummies(train['OverallCond'],prefix='overcond'))
	train['YearBuilt']=max(train['YearBuilt'])-train['YearBuilt']
	train['YearBuilt']=(train['YearBuilt']-train['YearBuilt'].mean())/train['YearBuilt'].std()
	train['YearRemodAdd']=max(train['YearRemodAdd'])-train['YearRemodAdd']
	train['YearRemodAdd']=(train['YearRemodAdd']-train['YearRemodAdd'].mean())/train['YearRemodAdd'].std()
	train=train.join(pd.get_dummies(train['RoofStyle'],prefix='roofstyle'))
	train=train.join(pd.get_dummies(train['RoofMatl'],prefix='roofmat'))
	train=train.join(pd.get_dummies(train['Exterior1st'],prefix='exterior1'))
	train=train.join(pd.get_dummies(train['Exterior2nd'],prefix='exterior2'))
	train=train.join(pd.get_dummies(train['MasVnrType'],prefix='masvnrtype'))
	train['MasVnrArea']=(train['MasVnrArea']-train['MasVnrArea'].mean())/train['MasVnrArea'].std()
	train=train.join(pd.get_dummies(train['ExterQual'],prefix='exterqual'))
	train=train.join(pd.get_dummies(train['ExterCond'],prefix='extercond'))
	train=train.join(pd.get_dummies(train['Foundation'],prefix='foundation'))
	train=train.join(pd.get_dummies(train['BsmtQual'],prefix='bsmtqual'))
	train=train.join(pd.get_dummies(train['BsmtCond'],prefix='bsmtcond'))
	train=train.join(pd.get_dummies(train['BsmtExposure'],prefix='bsmtexposure'))
	train=train.join(pd.get_dummies(train['BsmtFinType1'],prefix='bsmtfintype1'))
	train['BsmtFinSF1']=(train['BsmtFinSF1']-train['BsmtFinSF1'].mean())/train['BsmtFinSF1'].std()
	train=train.join(pd.get_dummies(train['BsmtFinType2'],prefix='bsmtfintype2'))
	train['BsmtFinSF2']=(train['BsmtFinSF2']-train['BsmtFinSF2'].mean())/train['BsmtFinSF2'].std()
	train['BsmtUnfSF']=(train['BsmtUnfSF']-train['BsmtUnfSF'].mean())/train['BsmtUnfSF'].std()
	

	
	train['TotalBsmtSF']=(train['TotalBsmtSF']-train['TotalBsmtSF'].mean())/train['TotalBsmtSF'].std()
	train=train.join(pd.get_dummies(train['Heating'],prefix='heating'))
	train=train.join(pd.get_dummies(train['HeatingQC'],prefix='heatingqc'))
	train=train.join(pd.get_dummies(train['CentralAir'],prefix='centralair'))
	train=train.join(pd.get_dummies(train['Electrical'],prefix='electrial'))
	train['1stFlrSF']=(train['1stFlrSF']-train['1stFlrSF'].mean())/train['1stFlrSF'].std()
	train['2ndFlrSF']=(train['2ndFlrSF']-train['2ndFlrSF'].mean())/train['2ndFlrSF'].std()
	train['LowQualFinSF']=(train['LowQualFinSF']-train['LowQualFinSF'].mean())/train['LowQualFinSF'].std()
	train['GrLivArea']=(train['GrLivArea']-train['GrLivArea'].mean())/train['GrLivArea'].std()
	train=train.join(pd.get_dummies(train['KitchenQual'],prefix='KitchenQual'))
	train=train.join(pd.get_dummies(train['Functional'],prefix='functional'))
	train=train.join(pd.get_dummies(train['FireplaceQu'],prefix='fireplacequ'))
	train=train.join(pd.get_dummies(train['GarageType'],prefix='garagetype'))
	train['GarageYrBlt']=max(train['GarageYrBlt'])-train['GarageYrBlt']
	train['GarageYrBlt']=(train['GarageYrBlt']-train['GarageYrBlt'].mean())/train['GarageYrBlt'].std()
	train=train.join(pd.get_dummies(train['GarageFinish'],prefix='garagefinish'))
	train['GarageArea']=(train['GarageArea']-train['GarageArea'].mean())/train['GarageArea'].std()
	train=train.join(pd.get_dummies(train['GarageQual'],prefix='garagequal'))
	train=train.join(pd.get_dummies(train['GarageCond'],prefix='garagecond'))
	train=train.join(pd.get_dummies(train['PavedDrive'],prefix='pavedrive'))
	train['WoodDeckSF']=(train['WoodDeckSF']-train['WoodDeckSF'].mean())/train['WoodDeckSF'].std()
	train['OpenPorchSF']=(train['OpenPorchSF']-train['OpenPorchSF'].mean())/train['OpenPorchSF'].std()
	train['EnclosedPorch']=(train['EnclosedPorch']-train['EnclosedPorch'].mean())/train['EnclosedPorch'].std()
	train['3SsnPorch']=(train['3SsnPorch']-train['3SsnPorch'].mean())/train['3SsnPorch'].std()
	train['ScreenPorch']=(train['ScreenPorch']-train['ScreenPorch'].mean())/train['ScreenPorch'].std()
	train['PoolArea']=(train['PoolArea']-train['PoolArea'].mean())/train['PoolArea'].std()
	train=train.join(pd.get_dummies(train['PoolQC'],prefix='poolqc'))
	train=train.join(pd.get_dummies(train['Fence'],prefix='fence'))
	train=train.join(pd.get_dummies(train['MiscFeature'],prefix='miscfeature'))
	train['MiscVal']=(train['MiscVal']-train['MiscVal'].mean())/train['MiscVal'].std()
	train['MoSold']=(train['MoSold']-train['MoSold'].mean())/train['MoSold'].std()
	train['YrSold']=max(train['YrSold'])-train['YrSold']
	train['YrSold']=(train['YrSold']-train['YrSold'].mean())/train['YrSold'].std()
	train=train.join(pd.get_dummies(train['SaleType'],prefix='saletype'))
	train=train.join(pd.get_dummies(train['SaleCondition'],prefix='salecondition'))
	train=train.join(pd.get_dummies(train['TotRmsAbvGrd'],prefix='totrmsabvgrd'))
	train=train.join(pd.get_dummies(train['BedroomAbvGr'],prefix='bedroomabvgr'))
	train.drop(['MSSubClass','Street','MSZoning','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','BldgType',
		'HouseStyle','OverallQual','OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
		'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
		'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
		'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition','Alley','Condition2','Utilities','TotRmsAbvGrd','BedroomAbvGr','MasVnrArea'],axis=1,inplace=True)	
	#print train
	train.drop(train[train['Id']==1183].index)
	train.drop(train[train['Id']==692].index)
	print train
	
	
	
	test=train[train['Id']>=1461]
	train=train[train['Id']<1461]
	del train['Id']
	sub=test[['Id']]
	del test['Id']	
	
	
	"""
	#模型选择
	X_train,X_test,Y_train,Y_test=train_test_split(train,label,test_size=0.33)
	
	regs = [
    ['Lasso',Lasso()],
    ['LinearRegression',LinearRegression()],
    ['Ridge',Ridge()],
    ['ElasticNet',ElasticNet()],
    ['RANSACRegressor',RANSACRegressor()],
    ['HuberRegressor',HuberRegressor()],
    ['SVR',SVR(kernel='linear')],
    ['DecisionTreeRegressor',DecisionTreeRegressor()],
    ['ExtraTreeRegressor',ExtraTreeRegressor()],
    ['AdaBoostRegressor',AdaBoostRegressor(n_estimators=150)],
    ['ExtraTreesRegressor',ExtraTreesRegressor(n_estimators=150)],
    ['GradientBoostingRegressor',GradientBoostingRegressor(n_estimators=150)],
    ['RandomForestRegressor',RandomForestRegressor(n_estimators=150)],
    ['XGBRegressor',XGBRegressor(n_estimators=150)],
]
	
	preds=[]
	for reg_name,reg in regs:
		print reg_name
		reg.fit(X_train,Y_train)
		y_pred=reg.predict(X_test)
		if np.sum(y_pred<0)>0:
			print 'y_pred have '+str(np.sum(y_pred<0))+" are negtive, we replace it witt median value of y_pred"
			y_pred[y_pred<0]=np.median(y_pred)
		score=np.sqrt(mean_squared_error(np.log(y_pred),np.log(Y_test)))
		print 
		preds.append([reg_name,y_pred])
		
	final_results=[]
	for comb_len in range(1,len(regs)+1):
		print "Model num:"+str(comb_len)
		results=[]
		for comb in itertools.combinations(preds,comb_len):
			#选取一个模型的组合，比如comb_len=2的时候，comb为(['Lasso',y_pred],['Ridge',y_pred]
			pred_sum=0
			model_name=[]
			for reg_name,pre in comb:
				pred_sum+=pre
				model_name.append(reg_name)
			pred_sum/=comb_len
			model_name='+'.join(model_name)
			score=np.sqrt(mean_squared_error(np.log(pred_sum),np.log(Y_test)))
			results.append([model_name,score])
		#操作每一个融合模型的分数
		results=sorted(results,key=lambda x:x[1])
		for model_name,score in results:
			print model_name+":"+str(score)
		print 
		final_results.append(results[0])
		
		
	print "best set of models"
	print 
	for i in final_results:
		print i
	"""
		
		
	
	#选择模型，写入文件
	result=0
	choose_model=[GradientBoostingRegressor(n_estimators=150)]
	for model in choose_model:
		reg=model.fit(train,label)
		pre=reg.predict(test)
		result+=pre

	
	sub['SalePrice']=result
	list=[[int(x[0]),x[1]] for x in sub.values]
	with open("result.csv",'wb') as f:
		writer=csv.writer(f)
		writer.writerow(['Id','SalePrice'])
		for i in range(len(list)):
			writer.writerow(list[i])
	
		
		
		
	
	"""
	sub=test[['Id']]
	del test['Id']
	result=clf.predict(test)
	sub['SalePrice']=result
	list=[[int(x[0]),x[1]] for x in sub.values]
	with open("result.csv",'wb') as f:
		writer=csv.writer(f)
		writer.writerow(['Id','SalePrice'])
		for i in range(len(list)):
			writer.writerow(list[i])
	"""
	
	
	#calScore(result,label)
	#print train.head(3)
	#for i in train.loc[0]:
	#	print i

	#print clf
	
	#print set(train['Neighborhood'])
	#print train.head(5)
	#print len(train.columns)

def main():
	loadData()

if __name__=='__main__':
	main()
