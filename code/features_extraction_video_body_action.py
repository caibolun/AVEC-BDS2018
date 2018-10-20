#!/usr/bin/env python
# -*-coding:utf-8-*-
#author: zhao.yinhu
'''
Iterating over all the video,it may take more time
'''

import numpy as np 
import cv2
import os,sys
import pandas as pd

def main(file_path):
	lamda=0.15
	if not file_path.endswith('/'):
		file_path+='/'
	file_list=os.listdir(file_path)
	file_list=np.sort(file_list)
	features_data=[]

	for i in file_list:
		#iterating over all the video	
		cap=cv2.VideoCapture(file_path+i)
		print(file_path+i)
		print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		action_features=[]
		action_features.append(i)
		total_diff_frame=[]	
		ret,last_frame=cap.read()
		gray_last_frame=cv2.cvtColor(last_frame,cv2.COLOR_BGR2GRAY)
		last_p=0
		
		pause=0

		for j in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
			#get a frame
			ret,current_frame=cap.read()
			gray_current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
			grayAbsDiff=cv2.absdiff(gray_current_frame,gray_last_frame)
			#if len(grayAbsDiff[grayAbsDiff==0])<len(grayAbsDiff[grayAbsDiff!=0]):
			total_diff_frame.append(grayAbsDiff)	
			#print(grayAbsDiff)
			#print(gray_current_frame)
			cur_p=lamda*last_p+(1.0-lamda)*grayAbsDiff
			#print(cur_p)
			last_p=cur_p
			gray_last_frame=gray_current_frame
			#show a frame	
			cv2.imshow("grayAbsDiff",grayAbsDiff)
			cv2.imshow("gray_current_frame",gray_current_frame)
			if cv2.waitKey(10)&0xFF==ord('q'):
				break
			pause+=1
			if j==int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-250:
			#if j==2500:
				break
		#print(len(total_diff_frame)/(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-250))
		'''feature1'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/2
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		action_feature1=len(diff_frame)/(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-250)
		action_features.append(action_feature1)

		'''feature2'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/3
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<2*len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		action_feature2=len(diff_frame)/(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-250)
		action_features.append(action_feature2)

		'''feature3'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 2/3
		diff_frame=[]
		for k in total_diff_frame:
			if 2*len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		action_feature3=len(diff_frame)/(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-250)
		action_features.append(action_feature3)

		'''feature4'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/2
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		diff_frame_ave=[]
		for m in diff_frame:
			diff_frame_ave.append(np.mean(m))
		action_feature4=np.mean(diff_frame_ave)
		action_features.append(action_feature4)

		'''feature5'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/2
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		diff_frame_ave=[]
		for m in diff_frame:
			diff_frame_ave.append(np.mean(m))
		action_feature5=np.var(diff_frame_ave)
		action_features.append(action_feature5)

		'''feature6'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/2
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		diff_frame_ave=[]
		for m in diff_frame:
			diff_frame_ave.append(len(m[m!=0]))
		action_feature6=np.mean(diff_frame_ave)
		action_features.append(action_feature6)

		'''feature7'''
		#The ratio of pixels that are not 0 in each frame in the video is at least 1/2
		diff_frame=[]
		for k in total_diff_frame:
			if len(k[k==0])<len(k[k!=0]):
				diff_frame.append(k)
		#print(len(diff_frame))
		diff_frame_ave=[]
		for m in diff_frame:
			diff_frame_ave.append(len(m[m!=0]))
		action_feature7=np.var(diff_frame_ave)
		action_features.append(action_feature7)

		cap.release()
		cv2.destroyAllWindows()

		features_data.append(action_features)

		features=['instance_name','action_feature1','action_feature2','action_feature3','action_feature4','action_feature5',\
				'action_feature6','action_feature7']

	#to DataFrame
	features_DataFrame=pd.DataFrame(features_data,columns=features)

	#save the feature file 
	features_DataFrame.to_csv("../features/body_action_features.csv",header=None,index=None)

if __name__=="__main__":
	file_path="../data/video"
	main(file_path)

