#!/usr/bin/env python
# -*-coding:utf-8-*-
#author: zhao.yinhu

import pandas as pd 
import numpy as np 
import os,sys

def main(file_path):
	timestamp_data=pd.read_csv("../data/time_3topics.csv",header=None)

	if not file_path.endswith('/'):
		file_path+='/'
	#Read all files
	file_list=os.listdir(file_path)
	file_list=np.sort(file_list)
	#data list
	features_data=[]
	timestamp_k=0
	for i in file_list:
		#Traverse all files
		dataset=pd.read_csv(file_path+i)
		gaze_features=[]
		head_features=[]
		gaze_features.append(i)

		#gaze data
		gaze_angle_x=np.array(dataset[' gaze_angle_x'])
		gaze_angle_y=np.array(dataset[' gaze_angle_y'])
		#head data
		pose_Tx=np.array(dataset[' pose_Tx'])
		pose_Ty=np.array(dataset[' pose_Ty'])
		pose_Tz=np.array(dataset[' pose_Tz'])
		#part1 data
		#gaze_angle_x_part1=np.array(dataset[' gaze_angle_x'][1:10])
		start=int(30*timestamp_data.iloc[3*timestamp_k,0])
		end=int(30*timestamp_data.iloc[3*timestamp_k,1])+2
		gaze_angle_x_part1=np.array(dataset[' gaze_angle_x'][start:end])
		gaze_angle_y_part1=np.array(dataset[' gaze_angle_y'][start:end])
		#part2 data
		start=int(30*timestamp_data.iloc[3*timestamp_k+1,0])
		end=int(30*timestamp_data.iloc[3*timestamp_k+1,1])+2
		gaze_angle_x_part2=np.array(dataset[' gaze_angle_x'][start:end])
		gaze_angle_y_part2=np.array(dataset[' gaze_angle_y'][start:end])
		#part3 data
		start=int(30*timestamp_data.iloc[3*timestamp_k+2,0])-2
		end=int(30*timestamp_data.iloc[3*timestamp_k+2,1])
		gaze_angle_x_part3=np.array(dataset[' gaze_angle_x'][start:])
		gaze_angle_y_part3=np.array(dataset[' gaze_angle_y'][start:])
		#######################Extract global features###########################################
		#Max-Min
		gaze_angle_x_jicha_total=np.max(gaze_angle_x)-np.min(gaze_angle_x)
		gaze_features.append(gaze_angle_x_jicha_total)
		gaze_angle_y_jicha_total=np.max(gaze_angle_y)-np.min(gaze_angle_y)
		gaze_features.append(gaze_angle_y_jicha_total)
		#Mean
		gaze_angle_x_mean_total=np.mean(gaze_angle_x)
		gaze_features.append(gaze_angle_x_mean_total)
		gaze_angle_y_mean_total=np.mean(gaze_angle_y)
		gaze_features.append(gaze_angle_y_mean_total)
		#Var
		gaze_angle_x_var_total=np.mean(gaze_angle_x)
		gaze_features.append(gaze_angle_x_var_total)
		gaze_angle_y_var_total=np.mean(gaze_angle_y)
		gaze_features.append(gaze_angle_y_var_total)
		#left-right
		gaze_angle_x_positive_total=len([j for j in gaze_angle_x if j>0])/len(gaze_angle_x)
		gaze_features.append(gaze_angle_x_positive_total)
		gaze_angle_x_negative_total=len([j for j in gaze_angle_x if j<0])/len(gaze_angle_x)
		gaze_features.append(gaze_angle_x_negative_total)
		gaze_angle_x_zero_total=len([j for j in gaze_angle_x if j==0])/len(gaze_angle_x)
		gaze_features.append(gaze_angle_x_zero_total)
		#up-down
		gaze_angle_y_positive_total=len([j for j in gaze_angle_y if j>0])/len(gaze_angle_y)
		gaze_features.append(gaze_angle_y_positive_total)
		gaze_angle_y_negative_total=len([j for j in gaze_angle_y if j<0])/len(gaze_angle_y)
		gaze_features.append(gaze_angle_y_negative_total)
		gaze_angle_y_zero_total=len([j for j in gaze_angle_y if j==0])/len(gaze_angle_y)
		gaze_features.append(gaze_angle_y_zero_total)
		#Covariance
		gaze_angle_x_cov_total=np.cov(gaze_angle_x)
		gaze_features.append(gaze_angle_x_cov_total)
		gaze_angle_y_cov_total=np.cov(gaze_angle_y)
		gaze_features.append(gaze_angle_y_cov_total)
		#################################extrace pose features##########################
		#var
		head_x_var_total=np.var(pose_Tx)
		head_features.append(head_x_var_total)
		head_y_var_total=np.var(pose_Ty)
		head_features.append(head_y_var_total)
		head_z_var_total=np.var(pose_Tz)
		head_features.append(head_z_var_total)
		########################################################################################
		#######################extract part1 features###############################################
		#Max-Min
		gaze_angle_x_jicha_part1=np.max(gaze_angle_x_part1)-np.min(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_jicha_part1)
		gaze_angle_y_jicha_part1=np.max(gaze_angle_y_part1)-np.min(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_jicha_part1)
		#Mean
		gaze_angle_x_mean_part1=np.mean(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_mean_part1)
		gaze_angle_y_mean_part1=np.mean(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_mean_part1)
		#Var
		gaze_angle_x_var_part1=np.mean(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_var_part1)
		gaze_angle_y_var_part1=np.mean(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_var_part1)
		#left-right
		gaze_angle_x_positive_part1=len([j for j in gaze_angle_x_part1 if j>0])/len(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_positive_part1)
		gaze_angle_x_negative_part1=len([j for j in gaze_angle_x_part1 if j<0])/len(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_negative_part1)
		gaze_angle_x_zero_part1=len([j for j in gaze_angle_x_part1 if j==0])/len(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_zero_part1)
		#up-right
		gaze_angle_y_positive_part1=len([j for j in gaze_angle_y_part1 if j>0])/len(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_positive_part1)
		gaze_angle_y_negative_part1=len([j for j in gaze_angle_y_part1 if j<0])/len(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_negative_part1)
		gaze_angle_y_zero_part1=len([j for j in gaze_angle_y_part1 if j==0])/len(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_zero_part1)
		#Covariance
		gaze_angle_x_cov_part1=np.cov(gaze_angle_x_part1)
		gaze_features.append(gaze_angle_x_cov_part1)
		gaze_angle_y_cov_part1=np.cov(gaze_angle_y_part1)
		gaze_features.append(gaze_angle_y_cov_part1)
		#################################pose##########################
		#方差
		head_x_var_part1=np.var(pose_Tx)
		head_features.append(head_x_var_part1)
		head_y_var_part1=np.var(pose_Ty)
		head_features.append(head_y_var_part1)
		head_z_var_part1=np.var(pose_Tz)
		head_features.append(head_z_var_part1)
		########################################################################################
		#######################提取part2特征###############################################
		#var
		gaze_angle_x_jicha_part2=np.max(gaze_angle_x_part2)-np.min(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_jicha_part2)
		gaze_angle_y_jicha_part2=np.max(gaze_angle_y_part2)-np.min(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_jicha_part2)
		#mean
		gaze_angle_x_mean_part2=np.mean(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_mean_part2)
		gaze_angle_y_mean_part2=np.mean(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_mean_part2)
		#var
		gaze_angle_x_var_part2=np.mean(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_var_part2)
		gaze_angle_y_var_part2=np.mean(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_var_part2)
		#left-right
		gaze_angle_x_positive_part2=len([j for j in gaze_angle_x_part2 if j>0])/len(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_positive_part2)
		gaze_angle_x_negative_part2=len([j for j in gaze_angle_x_part2 if j<0])/len(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_negative_part2)
		gaze_angle_x_zero_part2=len([j for j in gaze_angle_x_part2 if j==0])/len(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_zero_part2)
		#up-down
		gaze_angle_y_positive_part2=len([j for j in gaze_angle_y_part2 if j>0])/len(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_positive_part2)
		gaze_angle_y_negative_part2=len([j for j in gaze_angle_y_part2 if j<0])/len(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_negative_part2)
		gaze_angle_y_zero_part2=len([j for j in gaze_angle_y_part2 if j==0])/len(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_zero_part2)
		#Covariance
		gaze_angle_x_cov_part2=np.cov(gaze_angle_x_part2)
		gaze_features.append(gaze_angle_x_cov_part2)
		gaze_angle_y_cov_part2=np.cov(gaze_angle_y_part2)
		gaze_features.append(gaze_angle_y_cov_part2)
		#################################pose##########################
		#var
		head_x_var_part2=np.var(pose_Tx)
		head_features.append(head_x_var_part2)
		head_y_var_part2=np.var(pose_Ty)
		head_features.append(head_y_var_part2)
		head_z_var_part2=np.var(pose_Tz)
		head_features.append(head_z_var_part2)
		########################################################################################
		#######################part3###############################################
		#Max-Min
		gaze_angle_x_jicha_part3=np.max(gaze_angle_x_part3)-np.min(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_jicha_part3)
		gaze_angle_y_jicha_part3=np.max(gaze_angle_y_part3)-np.min(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_jicha_part3)
		#mena
		gaze_angle_x_mean_part3=np.mean(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_mean_part3)
		gaze_angle_y_mean_part3=np.mean(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_mean_part3)
		#var
		gaze_angle_x_var_part3=np.mean(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_var_part3)
		gaze_angle_y_var_part3=np.mean(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_var_part3)
		#left-right
		gaze_angle_x_positive_part3=len([j for j in gaze_angle_x_part3 if j>0])/len(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_positive_part3)
		gaze_angle_x_negative_part3=len([j for j in gaze_angle_x_part3 if j<0])/len(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_negative_part3)
		gaze_angle_x_zero_part3=len([j for j in gaze_angle_x_part3 if j==0])/len(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_zero_part3)
		#up-down
		gaze_angle_y_positive_part3=len([j for j in gaze_angle_y_part3 if j>0])/len(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_positive_part3)
		gaze_angle_y_negative_part3=len([j for j in gaze_angle_y_part3 if j<0])/len(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_negative_part3)
		gaze_angle_y_zero_part3=len([j for j in gaze_angle_y_part3 if j==0])/len(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_zero_part3)
		#Covariance
		gaze_angle_x_cov_part3=np.cov(gaze_angle_x_part3)
		gaze_features.append(gaze_angle_x_cov_part3)
		gaze_angle_y_cov_part3=np.cov(gaze_angle_y_part3)
		gaze_features.append(gaze_angle_y_cov_part3)
		#################################head pose##########################
		#var
		head_x_var_part3=np.var(pose_Tx)
		head_features.append(head_x_var_part3)
		head_y_var_part3=np.var(pose_Ty)
		head_features.append(head_y_var_part3)
		head_z_var_part3=np.var(pose_Tz)
		head_features.append(head_z_var_part3)
		########################################################################################
		gaze_head_features=gaze_features+head_features

		features_data.append(gaze_head_features)
		
		features=['instance_name','gaze_angle_x_jicha_total','gaze_angle_y_jicha_total','gaze_angle_x_mean_total','gaze_angle_y_mean_total','gaze_angle_x_var_total',\
			'gaze_angle_y_var_total','gaze_angle_x_positive_total','gaze_angle_x_negative_total','gaze_angle_x_zero_total','gaze_angle_y_positive_total',\
			'gaze_angle_y_negative_total','gaze_angle_y_zero_total','gaze_angle_x_cov_total','gaze_angle_y_cov_total',\
			'head_x_var_total','head_y_var_total','head_z_var_total',\
			'gaze_angle_x_jicha_part1','gaze_angle_y_jicha_part1','gaze_angle_x_mean_part1','gaze_angle_y_mean_part1','gaze_angle_x_var_part1','gaze_angle_y_var_part1',\
			'gaze_angle_x_positive_part1','gaze_angle_x_negative_part1','gaze_angle_x_zero_part1','gaze_angle_y_positive_part1','gaze_angle_y_negative_part1',\
			'gaze_angle_y_zero_part1','gaze_angle_x_cov_part1','gaze_angle_y_cov_part1',\
			'head_x_var_part1','head_y_var_part1','head_z_var_part1',\
			'gaze_angle_x_jicha_part2','gaze_angle_y_jicha_part2','gaze_angle_x_mean_part2','gaze_angle_y_mean_part2','gaze_angle_x_var_part2','gaze_angle_y_var_part2',\
			'gaze_angle_x_positive_part2','gaze_angle_x_negative_part2','gaze_angle_x_zero_part2','gaze_angle_y_positive_part2','gaze_angle_y_negative_part2',\
			'gaze_angle_y_zero_part2','gaze_angle_x_cov_part2','gaze_angle_y_cov_part2',\
			'head_x_var_part2','head_y_var_part2','head_z_var_part2',\
			'gaze_angle_x_jicha_part3','gaze_angle_y_jicha_part3','gaze_angle_x_mean_part3','gaze_angle_y_mean_part3','gaze_angle_x_var_part3','gaze_angle_y_var_part3',\
			'gaze_angle_x_positive_part3','gaze_angle_x_negative_part3','gaze_angle_x_zero_part3','gaze_angle_y_positive_part3','gaze_angle_y_negative_part3',\
			'gaze_angle_y_zero_part3','gaze_angle_x_cov_part3','gaze_angle_y_cov_part3',\
			'head_x_var_part3','head_y_var_part3','head_z_var_part3']
		timestamp_k+=1

	#DataFrame
	features_DataFrame=pd.DataFrame(features_data,columns=features)
	#part1
	part1_features=['instance_name','gaze_angle_x_jicha_part1','gaze_angle_y_jicha_part1','gaze_angle_x_mean_part1','gaze_angle_y_mean_part1','gaze_angle_x_var_part1','gaze_angle_y_var_part1',\
			'gaze_angle_x_positive_part1','gaze_angle_x_negative_part1','gaze_angle_x_zero_part1','gaze_angle_y_positive_part1','gaze_angle_y_negative_part1',\
			'gaze_angle_y_zero_part1','gaze_angle_x_cov_part1','gaze_angle_y_cov_part1',\
			'head_x_var_part1','head_y_var_part1','head_z_var_part1']
	#part2
	part2_features=['instance_name','gaze_angle_x_jicha_part2','gaze_angle_y_jicha_part2','gaze_angle_x_mean_part2','gaze_angle_y_mean_part2','gaze_angle_x_var_part2','gaze_angle_y_var_part2',\
			'gaze_angle_x_positive_part2','gaze_angle_x_negative_part2','gaze_angle_x_zero_part2','gaze_angle_y_positive_part2','gaze_angle_y_negative_part2',\
			'gaze_angle_y_zero_part2','gaze_angle_x_cov_part2','gaze_angle_y_cov_part2',\
			'head_x_var_part2','head_y_var_part2','head_z_var_part2']
	#part3
	part3_features=['instance_name','gaze_angle_x_jicha_part3','gaze_angle_y_jicha_part3','gaze_angle_x_mean_part3','gaze_angle_y_mean_part3','gaze_angle_x_var_part3','gaze_angle_y_var_part3',\
			'gaze_angle_x_positive_part3','gaze_angle_x_negative_part3','gaze_angle_x_zero_part3','gaze_angle_y_positive_part3','gaze_angle_y_negative_part3',\
			'gaze_angle_y_zero_part3','gaze_angle_x_cov_part3','gaze_angle_y_cov_part3',\
			'head_x_var_part3','head_y_var_part3','head_z_var_part3']

	gaze_head_part1=features_DataFrame[part1_features]
	gaze_head_part2=features_DataFrame[part2_features]
	gaze_head_part3=features_DataFrame[part3_features]
	#save the features file
	#features_DataFrame.to_csv("../features/gaze_head_features.csv",index=None,header=None)
	gaze_head_part1.to_csv("../features/gaze_head_part1.csv",index=None,header=None)
	gaze_head_part2.to_csv("../features/gaze_head_part2.csv",index=None,header=None)
	gaze_head_part3.to_csv("../features/gaze_head_part3.csv",index=None,header=None)
	print("extract features successful!!!")
	return features_DataFrame,gaze_head_part1,gaze_head_part2,gaze_head_part3

if __name__=="__main__":
	file_path="../data/AU"
	features_DataFrame,gaze_head_part1,gaze_head_part2,gaze_head_part3=main(file_path)

