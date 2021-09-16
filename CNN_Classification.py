import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math as m
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import math
from tensorflow import keras
from tensorflow.keras import layers
from random import shuffle
from keras import backend as K 
import numpy as np
import keras.datasets
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from  sklearn import model_selection
import scipy
import time
from footprints_and_cutouts import preprocess_scenes


(train_cutouts_blended_allScenes, train_cutouts_unblended_allScenes,
  train_blended_mag_filters_allScenes, train_unblended_mag_filters_allScenes) = preprocess_scenes(train=True,use_pipeline_segmap=True)

(test_cutouts_blended_allScenes, test_cutouts_unblended_allScenes,
 test_blended_mag_filters_allScenes, test_unblended_mag_filters_allScenes) = preprocess_scenes(train=False,use_pipeline_segmap=True)

train_cutouts_allScenes = []
train_cutouts_labels = []
train_mag_filter = []
count=0
for i in train_cutouts_unblended_allScenes:
	train_cutouts_allScenes.append(i)
	train_cutouts_labels.append([1,0])
	if train_unblended_mag_filters_allScenes[21.5][count]:
	    train_mag_filter.append(21.5)
	elif train_unblended_mag_filters_allScenes[22.5][count]:
	    train_mag_filter.append(22.5)
	elif train_unblended_mag_filters_allScenes[23.5][count]:
	    train_mag_filter.append(23.5)
	elif train_unblended_mag_filters_allScenes[24.5][count]:
	    train_mag_filter.append(24.5)
	elif train_unblended_mag_filters_allScenes[25.5][count]:
	    train_mag_filter.append(25.5)
	elif train_unblended_mag_filters_allScenes[26.5][count]:
	    train_mag_filter.append(26.5)
	else:
	    train_mag_filter.append(0)
	count+=1

count = 0
for i in train_cutouts_blended_allScenes:
	train_cutouts_allScenes.append(i)
	train_cutouts_labels.append([0,1])
	if train_blended_mag_filters_allScenes[21.5][count]:
	    train_mag_filter.append(21.5)
	elif train_blended_mag_filters_allScenes[22.5][count]:
	    train_mag_filter.append(22.5)
	elif train_blended_mag_filters_allScenes[23.5][count]:
	    train_mag_filter.append(23.5)
	elif train_blended_mag_filters_allScenes[24.5][count]:
	    train_mag_filter.append(24.5)
	elif train_blended_mag_filters_allScenes[25.5][count]:
	    train_mag_filter.append(25.5)
	elif train_blended_mag_filters_allScenes[26.5][count]:
	    train_mag_filter.append(26.5)
	else:
	    train_mag_filter.append(0)
	count+=1

test_cutouts_allScenes = []
test_cutouts_labels = []
test_mag_filter = []
count=0
for i in test_cutouts_unblended_allScenes:
	test_cutouts_allScenes.append(i)
	test_cutouts_labels.append([1,0])
	if test_unblended_mag_filters_allScenes[21.5][count]:
	    test_mag_filter.append(21.5)
	elif test_unblended_mag_filters_allScenes[22.5][count]:
	    test_mag_filter.append(22.5)
	elif test_unblended_mag_filters_allScenes[23.5][count]:
	    test_mag_filter.append(23.5)
	elif test_unblended_mag_filters_allScenes[24.5][count]:
	    test_mag_filter.append(24.5)
	elif test_unblended_mag_filters_allScenes[25.5][count]:
	    test_mag_filter.append(25.5)
	elif test_unblended_mag_filters_allScenes[26.5][count]:
	    test_mag_filter.append(26.5)
	else:
	    test_mag_filter.append(0)
	count+=1

count = 0
for i in test_cutouts_blended_allScenes:
	test_cutouts_allScenes.append(i)
	test_cutouts_labels.append([0,1])
	if test_blended_mag_filters_allScenes[21.5][count]:
	    test_mag_filter.append(21.5)
	elif test_blended_mag_filters_allScenes[22.5][count]:
	    test_mag_filter.append(22.5)
	elif test_blended_mag_filters_allScenes[23.5][count]:
	    test_mag_filter.append(23.5)
	elif test_blended_mag_filters_allScenes[24.5][count]:
	    test_mag_filter.append(24.5)
	elif test_blended_mag_filters_allScenes[25.5][count]:
	    test_mag_filter.append(25.5)
	elif test_blended_mag_filters_allScenes[26.5][count]:
	    test_mag_filter.append(26.5)
	else:
	    test_mag_filter.append(0)
	count+=1
for _ in np.arange(23):

	trainx,testx,trainy,testy,trainmag,testmag = train_cutouts_allScenes,test_cutouts_allScenes,train_cutouts_labels,test_cutouts_labels,train_mag_filter,test_mag_filter
	trainx2 = np.log10(np.array(trainx)+10**-8)
	testx2 = np.log10(np.array(testx)+10**-8)
	trainxnorm = (trainx2 - np.min(trainx2))/(np.max(trainx2)-np.min(trainx2))
	testxnorm = (testx2 - np.min(testx2))/(np.max(testx2)-np.min(testx2))

	input_shape = (23, 23, 1)
	num_classes=2
	model = keras.Sequential()

	model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(800,activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(400,activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(200,activation = 'relu'))
	model.add(Dense(num_classes, activation="softmax"))
	epochs=20
	model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
	time_start = time.time()
	model.fit(np.reshape(trainxnorm,(len(trainx),23,23,1)), np.array(trainy), epochs=15,verbose=True,batch_size=200,validation_split = .1)
	train_time = time.time()-time_start
	mean_loss_xxx = model.evaluate(np.array(np.reshape(testxnorm,(len(testx),23,23,1))),np.array(testy))

	bce=[]
	count = 0
	for i in testxnorm: 
	    bce.append(model.evaluate(np.array(np.reshape(i,(1,23,23,1))),np.array([testy[count]]))[0])
	    count+=1

	standard_error_bce = scipy.stats.sem(bce)

	classified = []
	blended_predict = []
	count = []
	for i in model.predict(np.reshape(testxnorm,(len(testx),23,23,1))):
	    if i[0]>i[1]:
	        classified.append([1,0])
	    else:
	        classified.append([0,1])
	    blended_predict.append(i[1])
	blended_predict
	        
	    
	arr_21 = []
	arr_22 = []
	arr_23 = []
	arr_24 = []
	arr_25 = []
	arr_26 = []
	count = 0
	for i in np.array(testy)==classified:
	    if testmag[count] == 21.5:
	        arr_21.append([i[0],testy[count]])
	    if testmag[count] == 22.5:
	        arr_22.append([i[0],testy[count]])
	    if testmag[count] == 23.5:
	        arr_23.append([i[0],testy[count]])
	    if testmag[count] == 24.5:
	        arr_24.append([i[0],testy[count]])
	    if testmag[count] == 25.5:
	        arr_25.append([i[0],testy[count]])
	    if testmag[count] == 26.5:
	        arr_26.append([i[0],testy[count]])
	    count+=1
	    
	arr_21_unblended = []
	arr_21_blended = []
	for i in arr_21:
	    if i[1] == [1,0]:
	        arr_21_unblended.append(i[0])
	    else:
	        arr_21_blended.append(i[0])
	arr_22_unblended = []
	arr_22_blended = []
	for i in arr_22:
	    if i[1] == [1,0]:
	        arr_22_unblended.append(i[0])
	    else:
	        arr_22_blended.append(i[0])
	arr_23_unblended = []
	arr_23_blended = []
	for i in arr_23:
	    if i[1] == [1,0]:
	        arr_23_unblended.append(i[0])
	    else:
	        arr_23_blended.append(i[0])
	arr_24_unblended = []
	arr_24_blended = []
	for i in arr_24:
	    if i[1] == [1,0]:
	        arr_24_unblended.append(i[0])
	    else:
	        arr_24_blended.append(i[0])
	arr_25_unblended = []
	arr_25_blended = []
	for i in arr_25:
	    if i[1] == [1,0]:
	        arr_25_unblended.append(i[0])
	    else:
	        arr_25_blended.append(i[0])
	arr_26_unblended = []
	arr_26_blended = []
	for i in arr_26:
	    if i[1] == [1,0]:
	        arr_26_unblended.append(i[0])
	    else:
	        arr_26_blended.append(i[0])
	unblended = [['accuracy','# of samples', 'variance of # of accurately classified samples'],[np.count_nonzero(arr_21_unblended)/len(arr_21_unblended),len(arr_21_unblended)],
	 [np.count_nonzero(arr_22_unblended)/len(arr_22_unblended),len(arr_22_unblended)],
	 [np.count_nonzero(arr_23_unblended)/len(arr_23_unblended),len(arr_23_unblended)],
	 [np.count_nonzero(arr_24_unblended)/len(arr_24_unblended),len(arr_24_unblended)],
	 [np.count_nonzero(arr_25_unblended)/len(arr_25_unblended),len(arr_25_unblended)],
	 [np.count_nonzero(arr_26_unblended)/len(arr_26_unblended),len(arr_26_unblended)]]
	blended = [['accuracy','# of samples', 'variance of # of accurately classified samples'],[np.count_nonzero(arr_21_blended)/len(arr_21_blended),len(arr_21_blended)],
	 [np.count_nonzero(arr_22_blended)/len(arr_22_blended),len(arr_22_blended)],
	 [np.count_nonzero(arr_23_blended)/len(arr_23_blended),len(arr_23_blended)],
	 [np.count_nonzero(arr_24_blended)/len(arr_24_blended),len(arr_24_blended)],
	 [np.count_nonzero(arr_25_blended)/len(arr_25_blended),len(arr_25_blended)],
	 [np.count_nonzero(arr_26_blended)/len(arr_26_blended),len(arr_26_blended)]]
	 
	for i in unblended[1:]:
	    i.append(np.sqrt(i[0]*i[1]*(1-i[0])))
	for i in blended[1:]:
	    i.append(np.sqrt(i[0]*i[1]*(1-i[0])))
	unblended = np.array(unblended)
	blended = np.array(blended)

	overall_blended_accuracy = np.sum(i[0].astype(float)*i[1].astype(float) for i in blended[1:].astype(float))/np.sum(i[1].astype(float) for i in blended[1:].astype(float))
	overall_unblended_accuracy = np.sum(i[0].astype(float)*i[1].astype(float) for i in unblended[1:].astype(float))/np.sum(i[1].astype(float) for i in unblended[1:].astype(float))
	 
	blended_predict_0 = []
	blended_predict_1 = []
	blended_predict_2 = []
	blended_predict_3 = []
	blended_predict_4 = []
	blended_predict_5 = []
	blended_predict_6 = []
	blended_predict_7 = []
	blended_predict_8 = []
	blended_predict_9 = []
	count = 0
	for i in blended_predict:
	    if i <.1:
	        blended_predict_0.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.1 and i<.2:
	        blended_predict_1.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.2 and i<.3:
	        blended_predict_2.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.3 and i<.4:
	        blended_predict_3.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.4 and i<.5:
	        blended_predict_4.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.5 and i<.6:
	        blended_predict_5.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.6 and i<.7:
	        blended_predict_6.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.7 and i<.8:
	        blended_predict_7.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.8 and i<.9:
	        blended_predict_8.append([[0,1]==testy[count],blended_predict[count]])
	    if i >=.9:
	        blended_predict_9.append([[0,1]==testy[count],blended_predict[count]])
	    count+=1
	    
	blended_predict_0 = np.array(blended_predict_0)
	blended_predict_1 = np.array(blended_predict_1)
	blended_predict_2 = np.array(blended_predict_2)
	blended_predict_3 = np.array(blended_predict_3)
	blended_predict_4 = np.array(blended_predict_4)
	blended_predict_5 = np.array(blended_predict_5)
	blended_predict_6 = np.array(blended_predict_6)
	blended_predict_7 = np.array(blended_predict_7)
	blended_predict_8 = np.array(blended_predict_8)
	blended_predict_9 = np.array(blended_predict_9)

	cal_0 = np.count_nonzero(blended_predict_0[:,0])/len(blended_predict_0)
	cal_1 = np.count_nonzero(blended_predict_1[:,0])/len(blended_predict_1)
	cal_2 = np.count_nonzero(blended_predict_2[:,0])/len(blended_predict_2)
	cal_3 = np.count_nonzero(blended_predict_3[:,0])/len(blended_predict_3)
	cal_4 = np.count_nonzero(blended_predict_4[:,0])/len(blended_predict_4)
	cal_5 = np.count_nonzero(blended_predict_5[:,0])/len(blended_predict_5)
	cal_6 = np.count_nonzero(blended_predict_6[:,0])/len(blended_predict_6)
	cal_7 = np.count_nonzero(blended_predict_7[:,0])/len(blended_predict_7)
	cal_8 = np.count_nonzero(blended_predict_8[:,0])/len(blended_predict_8)
	cal_9 = np.count_nonzero(blended_predict_9[:,0])/len(blended_predict_9)

	mean_0 = np.mean(blended_predict_0[:,1])
	mean_1 = np.mean(blended_predict_1[:,1])
	mean_2 = np.mean(blended_predict_2[:,1])
	mean_3 = np.mean(blended_predict_3[:,1])
	mean_4 = np.mean(blended_predict_4[:,1])
	mean_5 = np.mean(blended_predict_5[:,1])
	mean_6 = np.mean(blended_predict_6[:,1])
	mean_7 = np.mean(blended_predict_7[:,1])
	mean_8 = np.mean(blended_predict_8[:,1])
	mean_9 = np.mean(blended_predict_9[:,1])


	percentile_0 = [np.percentile(blended_predict_0[:,1],16),np.percentile(blended_predict_0[:,1],84)]
	percentile_1 = [np.percentile(blended_predict_1[:,1],16),np.percentile(blended_predict_1[:,1],84)]
	percentile_2 = [np.percentile(blended_predict_2[:,1],16),np.percentile(blended_predict_2[:,1],84)]
	percentile_3 = [np.percentile(blended_predict_3[:,1],16),np.percentile(blended_predict_3[:,1],84)]
	percentile_4 = [np.percentile(blended_predict_4[:,1],16),np.percentile(blended_predict_4[:,1],84)]
	percentile_5 = [np.percentile(blended_predict_5[:,1],16),np.percentile(blended_predict_5[:,1],84)]
	percentile_6 = [np.percentile(blended_predict_6[:,1],16),np.percentile(blended_predict_6[:,1],84)]
	percentile_7 = [np.percentile(blended_predict_7[:,1],16),np.percentile(blended_predict_7[:,1],84)]
	percentile_8 = [np.percentile(blended_predict_8[:,1],16),np.percentile(blended_predict_8[:,1],84)]
	percentile_9 = [np.percentile(blended_predict_9[:,1],16),np.percentile(blended_predict_9[:,1],84)]

	errors = np.array([percentile_0-mean_0,percentile_1-mean_1,percentile_2-mean_2,percentile_3-mean_3,percentile_4-mean_4,percentile_5-mean_5,
	              percentile_6-mean_6,percentile_7-mean_7,percentile_8-mean_8,percentile_9-mean_9])
	yerr_0 = np.sqrt((cal_0*(1-cal_0)/len(blended_predict_0)))
	yerr_1 = np.sqrt((cal_1*(1-cal_1)/len(blended_predict_1)))
	yerr_2 = np.sqrt((cal_2*(1-cal_2)/len(blended_predict_2)))
	yerr_3 = np.sqrt((cal_3*(1-cal_3)/len(blended_predict_3)))
	yerr_4 = np.sqrt((cal_4*(1-cal_4)/len(blended_predict_4)))
	yerr_5 = np.sqrt((cal_5*(1-cal_5)/len(blended_predict_5)))
	yerr_6 = np.sqrt((cal_6*(1-cal_6)/len(blended_predict_6)))
	yerr_7 = np.sqrt((cal_7*(1-cal_7)/len(blended_predict_7)))
	yerr_8 = np.sqrt((cal_8*(1-cal_8)/len(blended_predict_8)))
	yerr_9 = np.sqrt((cal_9*(1-cal_9)/len(blended_predict_9)))

	cal_data = [['# of samples', 'Model Blend Probability','Blend Fractions','xerr','yerr']]
	lens = [len(blended_predict_0),len(blended_predict_1),len(blended_predict_2),len(blended_predict_3),len(blended_predict_4),
	        len(blended_predict_5),len(blended_predict_6),len(blended_predict_7),len(blended_predict_8),len(blended_predict_9)]
	means = [mean_0,mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7,mean_8,mean_9]
	cals = [cal_0,cal_1,cal_2,cal_3,cal_4,cal_5,cal_6,cal_7,cal_8,cal_9]
	xerr = abs(np.array(errors))
	yerr = [yerr_0,yerr_1,yerr_2,yerr_3,yerr_4, yerr_5,yerr_6,yerr_7,yerr_8,yerr_9]

	cal_data.append(lens)
	cal_data.append(means)
	cal_data.append(cals)
	cal_data.append(xerr)
	cal_data.append(yerr) 

	np.savez("pipeline_data/classification_stats_"+str(time.time()),overall_blended_accuracy = overall_blended_accuracy,
	        overall_unblended_accuracy = overall_unblended_accuracy,train_time = train_time,bce = bce, sem_bce = standard_error_bce,unblended=unblended,blended=blended,calibration=cal_data)  
	                   
