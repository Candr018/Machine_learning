#!/usr/bin/env python
# coding: utf-8

# In[84]:


import csv
import random
import math
import pandas
import operator
import numpy 

def train(trainX, trainY, epochs, regularization, learning_rate, y_int):
    w = numpy.zeros((trainX.shape[1],1))
    numpy.append(w, y_int)
    for x in range(epochs):
        print(x)
        for i in range(trainX.shape[0]):
             if trainY[i]*numpy.dot(w.T,trainX[i,:]) < 1:
                dw = trainY[i] * numpy.expand_dims(trainX[i,:],axis=-1) + 2 * regularization * w
                w = w - (dw * learning_rate)
    return w   


# In[68]:


def test(testX, testY, w):
    correct = 0.
    for i in range(testX.shape[0]):
        if testY[i]*numpy.dot(w.T,testX[i,:]) > 0:
            correct += 1
    return correct/testX.shape[0]


# In[62]:


train_dataset = pandas.read_csv("mnist_train.csv") #loading CSV mnist training set
test_dataset = pandas.read_csv("mnist_test.csv")   #loading CSV mnist testing set
   
df1 = pandas.DataFrame(train_dataset)              #define the training dataset as a dataframe in pandas to manip data
df2 = pandas.DataFrame(test_dataset)               #define the testing dataset as a dataframe in pandas to manip data
split_correct = 0.0
    
#parcing data between lables (0-9) and image data(60000,785) for training set
train_data = df1.iloc[:,1:785]                     #training data
train_labels = df1.iloc[:,0]                       #labels for training data 
y = train_labels.to_numpy()                        #casting the lables as numpy array to perform calculations
x = train_data.to_numpy()                          #casting the data as a 2D numpy array to perform calculations


# In[63]:


index = numpy.arange(len(x))
numpy.random.shuffle(index)
    


# In[64]:


#chunk1 = numpy.empty((10000,784), dtype = numpy.int8);

#test1 = numpy.empty((10000,), dtype = numpy.int8);

chunk1= x[index[0:10000],:]
chunk2= x[index[10000:20000],:]
chunk3= x[index[20000:30000],:]
chunk4= x[index[30000:40000],:]
chunk5= x[index[40000:50000],:]
chunk6= x[index[50000:60000],:]

test1 = y[index[0:10000]]
test2 = y[index[10000:20000]]
test3 = y[index[20000:30000]]
test4 = y[index[30000:40000]]
test5 = y[index[40000:50000]]
test6 = y[index[50000:60000]]

    


# In[111]:


fold1_train = numpy.concatenate((chunk2, chunk3, chunk4, chunk5, chunk6), axis=0)
fold2_train = numpy.concatenate((chunk1, chunk3, chunk4, chunk5, chunk6), axis=0) 
fold3_train = numpy.concatenate((chunk1, chunk2, chunk4, chunk5, chunk6), axis=0)
fold4_train = numpy.concatenate((chunk1, chunk2, chunk3, chunk5, chunk6), axis=0)
fold5_train = numpy.concatenate((chunk1, chunk2, chunk3, chunk4, chunk6), axis=0)
fold6_train = numpy.concatenate((chunk1, chunk2, chunk3, chunk4, chunk5), axis=0)

fold1_label = numpy.concatenate((test2, test3, test4, test5, test6), axis=0)
fold2_label = numpy.concatenate((test1, test3, test4, test5, test6), axis=0) 
fold3_label = numpy.concatenate((test1, test2, test4, test5, test6), axis=0)
fold4_label = numpy.concatenate((test1, test2, test3, test5, test6), axis=0)
fold5_label = numpy.concatenate((test1, test2, test3, test4, test6), axis=0)
fold6_label = numpy.concatenate((test1, test2, test3, test4, test5), axis=0)

fold1_test = chunk1
fold2_test = chunk2
fold3_test = chunk3
fold4_test = chunk4
fold5_test = chunk5
fold6_test = chunk6

fold1_verify = test1
fold2_verify = test2
fold3_verify = test3
fold4_verify = test4
fold5_verify = test5
fold6_verify = test6

Ytrain = numpy.empty(fold1_label.shape, dtype = numpy.int8)
Ytest = numpy.empty(fold1_verify.shape, dtype = numpy.int8)

#EDIT HERE TO CHANGE
epochs = 5                                       #initalize the number of epochs for the SVM
alpha = 0.98                                     #initalize learning rate    
train_for = 1                                    #number to train for in mnist
b = 1                                            #y_int
for i in range(len(fold1_verify)):
    if fold1_verify[i] == train_for:
        Ytest[i] = 1
    else:
        Ytest[i] = -1
        
    if fold2_verify[i] == train_for:
        fold2_verify[i] = 1
    else:
        fold2_verify[i] = -1
        
    if fold3_verify[i] == train_for:
        fold3_verify[i] = 1
    else:
        fold3_verify[i] = -1
        
    if fold4_verify[i] == train_for:
        fold4_verify[i] = 1
    else:
        fold4_verify[i] = -1
        
    if fold5_verify[i] == train_for:
        fold5_verify[i] = 1
    else:
        fold5_verify[i] = -1
        
    if fold6_verify[i] == train_for:
        fold6_verify[i] = 1
    else:
        fold6_verify[i] = -1

for i in range(len(fold1_label)):
    if fold1_label[i] == train_for:
        Ytrain[i] = 1
    else:
        Ytrain[i] = -1
        
    if fold2_label[i] == train_for:
        fold2_label[i] = 1
    else:
        fold2_label[i] = -1
        
    if fold3_label[i] == train_for:
        fold3_label[i] = 1
    else:
        fold3_label[i] = -1
        
    if fold4_label[i] == train_for:
        fold4_label[i] = 1
    else:
        fold4_label[i] = -1
        
    if fold5_label[i] == train_for:
        fold5_label[i] = 1
    else:
        fold5_label[i] = -1
        
    if fold6_label[i] == train_for:
        fold6_label[i] = 1
    else:
        fold6_label[i] = -1


# In[112]:


#initalize test for regularization parameters
lamda = [100,10,1,0.1,0.01,0.001]
Ytrain = numpy.append(Ytrain, 0)
#folds = [numpy.append(fold1_train,0), fold2_train, fold3_train, fold4_train, fold5_train, fold6_train]

#foldslabels = [numpy.append(fold1_label,0), fold2_label, fold3_label, fold4_label, fold5_label, fold6_label]

#foldstest = [fold1_test, fold2_test, fold3_test, fold4_test, fold5_test, fold6_test]

#foldsverify = [fold1_verify, fold2_verify, fold3_verify, fold4_verify, fold5_verify, fold6_verify]

#split_correct = numpy.empty((6), dtype = numpy.float)
#print(lamda[1])
#w = train(fold1_train, Ytrain, epochs, lamda[1], alpha, b)      #determining our weights
#split_correct = test(fold1_test, Ytest, w)
#print(split_correct * 100)
w2 = train(fold1_train, Ytrain, epochs, lamda[2], alpha, b)      #determining our weights
split_correct = test(fold1_test, Ytest, w2)
print(split_correct * 100)


# In[113]:


#w = train(folds[1], foldslabels[1], epochs, lamda[1], alpha, b)      #determining our weights
#split_correct[1] = test(foldstest[1], foldsverify[1], w)
#print("When Lamda is:", lamda[1] ,",the percent correct is: ", split_correct[1] * 100)


# In[ ]:




