#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:17:08 2017

@author: deeptichevvuri
"""


import numpy
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score

class SVMClassifier:

    def __init__(self):
        #train data set
        self.traindata = numpy.loadtxt("features.train.txt")
        self.traindatafeatures = self.traindata[:,1:3]
        self.trainlabels = self.traindata[:,0]
        # test data set
        self.testdata = numpy.loadtxt("features.test.txt")
        self.testdatafeatures = self.testdata[:,1:3]
        self.testlabels = self.testdata[:,0]
        self.G  = '\033[32m' # green
        self.B  = '\033[34m' # blue
        self.W  = '\033[0m'  # white (normal)
        self.xhigh=0
        self.xlow=0
        
        # 2.2 Polynomial kernels (1) from even versus all even=[0,2,4,6,8]
        even = [0,2,4,6,8]
        print(self.G+"Polunomial Kernels-Question 1: C = 0.01 and Q = 2, classifier with highest Etrain"+self.W)
        self.PolynomialKernels1And2(even)
        
         # 2.2 Polynomial kernels (2) from odd versus all even=[1,3,5,7,9]
        odd = [1,3,5,7,9]
        print(self.G+"\nPolunomial Kernels-Question 2: C = 0.01 and Q = 2, classifier with lowest Etrain"+self.W)
        self.PolynomialKernels1And2(odd)
        
          # 2.2 Polynomial kernels (3) Comparing the two selected classifiers from Problems 1 and 2,
        print(self.G+"\nPolunomial Kernels-Question 3: The closest difference between the number of support vectors of these two classifier selected"+self.W)
        self.PolynomialKernels3()
        
        # 2.2 Polynomial kernels (4 and 5) versus 5 
        self.PolynomialKernels4And5()

        # 2.3 Cross validation
        print(self.G+"\nCross Validation-Question 1 and 2: C with lowest Calculated Error"+self.W)
        self.CrossValidation()

        # 2.4 Gaussian Kernel
        print(self.G+"\nGaussian Kernel-Question 1 and 2: C with lowest Error (train and test)"+self.W)
        self.GaussianKernel()

        
    def PolynomialKernels1And2(self, xs):
        highestEtrain=0.00
        lowestEtrain=1.00
        errorstring=''
        finalerror=0.00
        xfinal=0
        for x in xs:
            #building the Support Vector Kernel for Polynomial classification
            classifier = svm.SVC(C=0.01, degree=2, kernel='poly', coef0=1, gamma=1)
            newlabels = self.XvsAll(x, numpy.copy(self.traindata), numpy.copy(self.trainlabels))
            newtraindata = self.traindatafeatures
            classifier.fit(newtraindata, newlabels)
            classification = classifier.predict(newtraindata)
            acc = metrics.accuracy_score(newlabels, classification)
            error = 1-acc
            print ("For {} versus all: Accuracy = {} and Error = {}".format(x,acc,error))
            if x%2 == 0:
                if error > highestEtrain:
                    highestEtrain=error
                    finalerror=error
                    xfinal=x
                    self.xhigh=x
                    errorstring="the classifier with highest Etrain"
            else:
                if error < lowestEtrain:
                    lowestEtrain=error
                    finalerror=error
                    xfinal=x
                    self.xlow=x
                    errorstring="the classifier with lowest Etrain"
        print(self.B+"{} is  {} versus all with Error {}".format(errorstring,xfinal,finalerror)+ self.W) 
        
    def PolynomialKernels3(self):
        vectorcountclassifier1 = self.GetClassifierVectorCount(self.xhigh)
        vectorcountclassifier2 = self.GetClassifierVectorCount(self.xlow)
        #get support vetor count fro eah class
        print(self.B+"The diference in count of Support vector between {} versus all and {} versus all is {} for both classes".format(self.xhigh,self.xlow,vectorcountclassifier1-vectorcountclassifier2)+self.W)
    
        
    def GetClassifierVectorCount(self, x):
        classifier = svm.SVC(C=0.01, degree=2, kernel='poly', coef0=1, gamma=1)
        newlabels = self.XvsAll(x, numpy.copy(self.traindata), numpy.copy(self.trainlabels))
        newtraindata = self.traindatafeatures
        classifier.fit(newtraindata, newlabels)
        #get support vetor count for each class
        return classifier.n_support_
    
    def XvsAll(self, one, traindata, labels):
        #Begin 1 vs all
        for i in range(0, len(labels)):
            l = labels[i]
            if l==one:
                labels[i] = 1
            else:
                labels[i] = -1
        return labels

    def OnevsOne(self, a, b, traindata, labels):
        #Begin 1 vs all
        to_delete = []
        for i in range(0, len(labels)):
            l = labels[i]
            if l==a or l==b:
                pass
            else:
                to_delete.append(i)
        traindata = numpy.delete(traindata, to_delete, axis=0)
        labels = numpy.delete(labels, to_delete, axis=0)
        return traindata, labels
    
    def PolynomialKernels4And5(self):
        print (self.G+"\nPolunomial Kernels-Question 4: 1 versus 5 classifier with Q = 2 and C ∈ {0.001,0.01,0.1,1}:"+self.W)
        for C in [0.0001, 0.001, 0.01, 0.1, 1]:
            self.PK4And5(1,5,2,C)

        print (self.G+"\nPolunomial Kernels-Question 5: 1 versus 5 classifier, comparing Q = 2 with Q = 5:"+self.W)
        for C in [0.0001, 0.001, 0.01, 0.1, 1]:
            for Q in [2,5]:
                self.PK4And5(1,5,Q,C)

    def PK4And5(self, a, b, Q, C):
        newtraindata, newlabels = self.OnevsOne(a,b, numpy.copy(self.traindatafeatures), numpy.copy(self.trainlabels))
        classifier = svm.SVC(C=C, degree=Q, kernel='poly', coef0=1, gamma=1)
        classifier.fit(newtraindata, newlabels)
        classification = classifier.predict(newtraindata)
        print (self.G+"Classifier Parameters: Q={} and C={}".format(Q,C)+self.W)
        acc = metrics.accuracy_score(newlabels, classification)
        error = 1-acc
        print ("For {} vs {} Train: Accuracy = {}. Error = {}".format(a,b,acc,error))
        newtestdata, newtestlabels = self.OnevsOne(a,b, numpy.copy(self.testdatafeatures), numpy.copy(self.testlabels))
        classification = classifier.predict(newtestdata)
        acc = metrics.accuracy_score(newtestlabels, classification)
        error = 1-acc
        print ("For {} vs {} Test, Accuracy = {}. Error = {}".format(a,b,acc,error))

        print ("Support vectors count:{}".format(classifier.n_support_))

    def CrossValidation(self):
        a = 1
        b = 5
        newtraindata, newlabels = self.OnevsOne(a,b, numpy.copy(self.traindatafeatures), numpy.copy(self.trainlabels))
        newtestdata, newtestlabels = self.OnevsOne(a,b, numpy.copy(self.testdatafeatures), numpy.copy(self.testlabels))
        Q = 2   
        for C in [0.0001, 0.001, 0.01, 0.1, 1]:
            print ("For Q={} C={}".format(Q,C))
            classifier = svm.SVC(C=C, degree=Q, kernel='poly', coef0=1, gamma=1)    
            classifier.fit(newtraindata, newlabels)
            e_cv_train_mean = 0
            e_cv_test_mean = 0
            for i in range(100):
                scores = cross_val_score(classifier, newtraindata, newlabels, cv=10)
                e_cv_train = 1-scores.mean()
                e_cv_train_mean+=e_cv_train
                scores = cross_val_score(classifier, newtestdata, newtestlabels, cv=10)
                e_cv_test = 1-scores.mean()
                e_cv_test_mean+=e_cv_test
            print ("EC on Training set is: {}".format(e_cv_train_mean/100))
            print ("EC on Test set is: {}".format(e_cv_test_mean/100))

    def GaussianKernel(self):
        a=1
        b=5
        lowestEtrain=1.00
        cEtrain=0
        cEtest=0
        lowestEtest=1.00
        newtraindata, newlabels = self.OnevsOne(a,b, numpy.copy(self.traindatafeatures), numpy.copy(self.trainlabels))
        newtestdata, newtestlabels = self.OnevsOne(a,b, numpy.copy(self.testdatafeatures), numpy.copy(self.testlabels))
        for C in [0.01, 1, 100, pow(10,4), pow(10,6)]:
            classifier = svm.SVC(C=C, kernel='rbf', gamma=1)
            classifier.fit(newtraindata, newlabels)
            classification = classifier.predict(newtraindata)
            print ("Gaussian Classifier Parameters:C={}".format(C))
            acc = metrics.accuracy_score(newlabels, classification)
            error = 1-acc
            print ("For {} vs {} Train, Accuracy = {}. Error = {}".format(a,b,acc,error))
            if error< lowestEtrain:
                lowestEtrain=error
                cEtrain=C
            newtestdata, newtestlabels = self.OnevsOne(a,b, numpy.copy(self.testdatafeatures), numpy.copy(self.testlabels))
            classification = classifier.predict(newtestdata)
            acc = metrics.accuracy_score(newtestlabels, classification)
            error = 1-acc
            print ("For {} vs {} Test, Accuracy is {}. Error = {}".format(a,b,acc,error))
            if error< lowestEtest:
                lowestEtest=error
                cEtest=C
        print(self.B+"C={} results in lowest Etrain, C={} results in lowest Etest".format(cEtrain,cEtest)+self.W)

classifier = SVMClassifier()