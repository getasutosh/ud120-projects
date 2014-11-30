#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.metrics import accuracy_score
from sklearn import svm

#clf = svm.SVC(kernel='linear')
clf = svm.SVC(C=10000.0, kernel='rbf')
t0 = time()
"""training time due to reduced sample size: .149s and prediction time: 1.14s but the overall
prediction accuracy goes down to 0.8845
Without reduction in sample size, the training time : 216.546 s and prediction time is 21.794 s
But the overall accuracy is increased to 0.9840

But after changing the kernel to rbf and c value to 10000, the overall time is reduced.
Training time: 132.83s
Prediction Time: 15.634s
Accuracy increased to: .9908
"""
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = clf.predict(features_test)
print "Prediction  time:", round(time()-t0, 3), "s"
accuracy = accuracy_score(pred, labels_test)
print accuracy

from scipy.stats import itemfreq
print itemfreq(pred)

#########################################################


