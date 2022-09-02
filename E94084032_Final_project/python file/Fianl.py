# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:09:46 2022

@author: win10
"""

import random
import pylab
import math
import numpy as np
import sklearn.linear_model #https://scikit-learn.org/stable/
               #https://en.wikipedia.org/wiki/Logistic_regression
#import statistics

def variance(X):
    """Assumes that X is a list of numbers.
       Returns the standard deviation of X"""
    mean = sum(X)/len(X)
    tot = 0.0
    for x in X:
        tot += (x - mean)**2
    return tot/len(X)
def stdDev(X):
    """Assumes that X is a list of numbers.
       Returns the standard deviation of X"""
    return variance(X)**0.5

class Passenger2(object): 
    def __init__ (self,c1,c2,c3, age, gender, survived): 
        self.featureVec = (c1,c2,c3, age, gender) 
        self.label = survived  

    def featureDist(self, other): 
        dist = 0.0 
        for i in range(len(self.featureVec)): 
            dist += abs(self.featureVec[i] - other.featureVec[i])**2 
        return dist**0.5 
    
    def cosine_similarity(self,other):
    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(self.featureVec)):
            x = self.featureVec[i]; y = other.featureVec[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

    def getc1(self): 
        return self.featureVec[0]
    
    def getc2(self): 
        return self.featureVec[1]
    
    def getc3(self): 
        return self.featureVec[2]

    def getage(self): 
        return self.featureVec[3]
    
    def getgender(self): 
        return self.featureVec[4]
    

    def getlabel(self): 
        return self.label 

    def getFeatures(self): 
        return self.featureVec 

    #def __str__ (self): 
    #    return str(self.getc1()) + ', ' + str(self.getage())+ ', ' + str(self.getgender())+ ', ' + str(self.getlabel())  


def getData2(filename):
    
    data = {}
    f = open(filename)
    next(f)
    line = f.readline() 
    data['c1'], data['c2'], data['c3'] = [], [], []
    data['age'], data['gender'] = [], [] 
    data['last name'],data['name'] = [], []
    data['survived']  = []
     
    while line != '':
        split = line.split(',')
        if int(split[0]) == 1:
            data['c1'].append(1)
            data['c2'].append(0)
            data['c3'].append(0)
        elif int(split[0]) == 2:
            data['c1'].append(0)
            data['c2'].append(1)
            data['c3'].append(0)
        elif int(split[0]) == 3:
            data['c1'].append(0)
            data['c2'].append(0)
            data['c3'].append(1)
        data['age'].append(float(split[1]))
        
        if split[2] == 'M':
            data['gender'].append(int(1))
            
        elif split[2] == 'F':
            data['gender'].append(int(0))
           
        data['survived'].append(int(split[3]))
        data['last name'].append(str(split[4]))
        data['name'].append(str(split[5][:-1])) 
       #remove \n
        line = f.readline()
    f.close()
    return data

def buildpassengerExamples2(fileName): 
    
    data  = getData2(fileName)
    examples = [] 
    for i in range(len(data['survived'])):
        a = Passenger2(data['c1'][i],data['c2'][i], data['c3'][i],  data['age'][i],data['gender'][i],data['survived'][i])
        examples.append(a)           
    return examples

def makeHist(data, bins, title, xLabel, yLabel):
    mean = sum(data)/len(data)
    std = stdDev(data)
    pylab.hist(data, bins, edgecolor='black',label ='Maximum Accuracies'+"\n"+"mean="+str(round(mean,2))+" SD="+str(round(std,2)))
    pylab.title(title)
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.legend()
    #pylab.annotate('Mean = ' + str(round(mean, 2)) +              '\nSD = ' + str(round(std, 2)), fontsize = 20,
    #          xy = (0.50, 0.75), xycoords = 'axes fraction')
    
def makeHist2(data, bins, title, xLabel, yLabel):
    mean = sum(data)/len(data)
    std = stdDev(data)
    pylab.hist(data, bins, edgecolor='black',label ='k Values for Maximum Accuracies'+"\n"+"mean="+str(round(mean,2))+" SD="+str(round(std,2)))
    pylab.title(title)
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    pylab.legend()
#______________________part 1_________________________

def sepratepassengerExamples(fileName): 
    data  = getData2(fileName)
    examples_F = []
    examples_M = []
    for i in range(len(data['survived'])):
        a = Passenger2(data['c1'][i],data['c2'][i], data['c3'][i],  data['age'][i],data['gender'][i],data['survived'][i])
        if a.getgender() == 1:
            examples_M.append(a) 
        else:
            examples_F.append(a)
    return examples_M,examples_F

m,f=sepratepassengerExamples('TitanicPassengers.txt')

Mcabin1 = 0
Mcabin1_sur = 0
Mcabin2 = 0
Mcabin2_sur = 0
Mcabin3 = 0
Mcabin3_sur = 0

for i in m:
    if i.getc1() == 1:
        Mcabin1+=1
        if i.getlabel() ==1:
            Mcabin1_sur+=1
            
    elif i.getc2() == 1:
        Mcabin2 +=1
        if i.getlabel() ==1:
             Mcabin2_sur+=1   
    else:
        Mcabin3 +=1
        if i.getlabel() ==1:
            Mcabin3_sur+=1

Fcabin1 = 0
Fcabin1_sur = 0
Fcabin2 = 0
Fcabin2_sur = 0
Fcabin3 = 0
Fcabin3_sur = 0

for i in f:
    if i.getc1() == 1:
        Fcabin1+=1
        if i.getlabel() ==1:
            Fcabin1_sur+=1
            
    elif i.getc2() == 1:
        Fcabin2 +=1
        if i.getlabel() ==1:
             Fcabin2_sur+=1   
    else:
        Fcabin3 +=1
        if i.getlabel() ==1:
            Fcabin3_sur+=1

print("_________part1_________")            
print("---------male----------")
print("male in cabin 1:",Mcabin1)
print("male in cabin 2:",Mcabin2)
print("male in cabin 3:",Mcabin3) 
print("male in cabin 1 survived:",Mcabin1_sur)
print("male in cabin 2 survived:",Mcabin2_sur)
print("male in cabin 3 survived:",Mcabin3_sur) 
print("---------female----------")
print("female in cabin 1:",Fcabin1)
print("female in cabin 2:",Fcabin2)
print("female in cabin 3:",Fcabin3) 
print("female in cabin 1 survived:",Fcabin1_sur)
print("female in cabin 2 survived:",Fcabin2_sur)
print("female in cabin 3 survived:",Fcabin3_sur)     
print("")


#________________________part 2_________________________

def seprate_M_F(examples):
    Male_age = []
    Female_age = []
    for e in examples:
        if e.getgender() == 1:
            Male_age.append(e.getage())
        elif e.getgender() == 0:
            Female_age.append(e.getage())
    return Male_age , Female_age

survived_male_age = []

survived_female_age = []
            
def choose_servived(examples):
    survived_age = []
    for e in examples:
        if e.getlabel()== 1 and e.getgender() == 1:
            survived_male_age.append(e.getage())
        elif e.getlabel()== 1 and e.getgender() == 0:
            survived_female_age.append(e.getage())
            
    return survived_male_age,survived_female_age

def choose_cabin(examples):
    Male_cabin =[]
    Male_cabin_sur =[]
    #Male_carbin3 =[]
    Female_cabin =[]
    Female_cabin_sur =[]
    #Female_carbin3 =[]
    for e in examples:
        if e.getgender() == 1:
            if e.getc1() == 1:
                Male_cabin.append(int(1))
            elif e.getc2() == 1:
                Male_cabin.append(int(2))
            elif e.getc3() == 1:
                Male_cabin.append(int(3))
            if e.getlabel() ==1:
                if e.getc1() == 1:
                    Male_cabin_sur.append(int(1))
                elif e.getc2() == 1:
                    Male_cabin_sur.append(int(2))
                elif e.getc3() == 1:
                    Male_cabin_sur.append(int(3))
        else:
            if e.getc1() == 1:
                Female_cabin.append(int(1))
            elif e.getc2() == 1:
                Female_cabin.append(int(2))
            elif e.getc3() == 1:
                Female_cabin.append(int(3))
            if e.getlabel() == 1:
                if e.getc1() == 1:
                    Female_cabin_sur.append(int(1))
                elif e.getc2() == 1:
                    Female_cabin_sur.append(int(2))
                elif e.getc3() == 1:
                    Female_cabin_sur.append(int(3))
                 
    return Male_cabin  ,Male_cabin_sur ,Female_cabin ,Female_cabin_sur  
            
                
            
            
  
                
                    
                    
    

x=buildpassengerExamples2('TitanicPassengers.txt')
Male_age , Female_age =seprate_M_F(x)
servived_male_age,servived_female_age = choose_servived(x)


pylab.hist(Male_age,20,label ='All male Passengers'+'\n'+'Mean='+ str(round(sum(Male_age)/len(Male_age),3)) + " SD="+str(round(stdDev(Male_age),3)), edgecolor='black')
pylab.hist(servived_male_age,20,label ='Survived male Passengers'+'\n'+'Mean='+ str(round(sum( servived_male_age)/len( servived_male_age),3)) + " SD="+str(round(stdDev( servived_male_age),3)), edgecolor='black')
pylab.title("Male Passengers and Survived")
pylab.xlabel("Male Ages")
pylab.ylabel("Number of Male passengers")
pylab.legend()
pylab.show()

#makeHist(Female_age,20,"","","")
#makeHist(servived_female_age ,20,"","","")
pylab.show()
pylab.hist(Female_age,20,label ='All Female Passengers'+'\n'+'Mean='+ str(round(sum(Female_age)/len(Female_age),3)) + " SD="+str(round(stdDev(Female_age),3)), edgecolor='black')
pylab.hist(servived_female_age,20,label ='Survived Female Passengers'+'\n'+'Mean='+ str(round(sum(servived_female_age)/len(servived_female_age),3)) + " SD="+str(round(stdDev( servived_female_age),3)), edgecolor='black')
pylab.title("Female Passengers and Survived")
pylab.xlabel("Female Ages")
pylab.ylabel("Number of Female passengers")
pylab.legend()
pylab.show()

Male_carbin  ,Male_carbin_ser ,Female_carbin ,Female_carbin_ser=choose_cabin(x)
 

pylab.hist(Male_carbin,3,range=[1,3], edgecolor='black')
pylab.hist(Male_carbin_ser,3,range=[1,3], edgecolor='black')
pylab.title('male cabin classes and survived')
pylab.xlabel('male cabin classes')
pylab.ylabel('number of male passengers')
pylab.xticks([1,2,3])
#pylab.hist(Male_carbin_ser, edgecolor='black')
pylab.show()


pylab.hist(Female_carbin,3,range=[1,3], edgecolor='black')
pylab.hist(Female_carbin_ser,3,range=[1,3], edgecolor='black')
pylab.title('female cabin classes and survived')
pylab.xlabel('female cabin classes')
pylab.ylabel('number of female passengers')
pylab.xticks([1,2,3])
#pylab.hist(Male_carbin_ser, edgecolor='black')
pylab.show()

#____________________part 3_____________________

def applyModel(model, testSet, label, prob = 0.5):
    #Create vector containing feature vectors for all test examples
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getlabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getlabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg



def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = False):
 
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
    sens = sensitivity(truePos, falseNeg) 
    spec = specificity(trueNeg, falsePos) 
    ppv = posPredVal(truePos, falsePos)
    
    
    if toPrint: 
        print(' Accuracy =', round(accur, 3)) 
        print(' Sensitivity =', round(sens, 3)) 
        print(' Specificity =', round(spec, 3)) 
        print(' Pos. Pred. Val. =', round(ppv, 3)) 
    return accur,sens,spec,ppv

def divide80_20_1000(examples):
    sampleIndices = random.sample(range(len(examples)), len(examples)//5) 
    trainingSet, testSet = [], [] 
    for i in range(len(examples)): 
        if i in sampleIndices: 
            testSet.append(examples[i]) 
        else: trainingSet.append(examples[i]) 
    return trainingSet, testSet



def accuracy(truePos, falsePos, trueNeg, falseNeg): 
    numerator = truePos + trueNeg 
    denominator = truePos + trueNeg + falsePos + falseNeg 
    return numerator/denominator 
def sensitivity(truePos, falseNeg): 
    try: 
        return truePos/(truePos + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 
def specificity(trueNeg, falsePos): 
    try: 
        return trueNeg/(trueNeg + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def posPredVal(truePos, falsePos): 
    try:
        return truePos/(truePos + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def negPredVal(trueNeg, falseNeg): 
    try: 
        return trueNeg/(trueNeg + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 
    

def buildROC(model, testSet, label, title, plot = True):
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg =\
                               applyModel(model, testSet, label, p)
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals)
    if plot:
        pylab.plot(xVals, yVals)
        pylab.plot([0,1], [0,1,], '--')
        pylab.title(title +  ' (AUROC = '\
                    + str(round(auroc, 3)) + ')')
        pylab.xlabel('1 - Specificity')
        pylab.ylabel('Sensitivity')
    return auroc


print("__________________part 3_________________")
Accr,Sens,Spec,Ppv = [],[],[],[]
c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]
#age = []
roc = []
MaxK = []
MaxA = []
mean_accur = [None]*201 
for i in range (1000):        
    trainset , testset = divide80_20_1000(x) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
    
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 601, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)

        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

    
#除以一千的原因為總共做一千次
print("Logistic Regression:") 
print("Averages for all examples 1000 trials with k=0.5")   
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Spec)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Spec),3)) 
print('Pos. Pred. Val. =', round(sum(Ppv)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Ppv),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,20,"Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 20, "Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 601, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

#------------------------------------------------------------------------------

#____________________part 4________________________
def zScaleFeatures(vals): 
    """Assumes vals is a sequence of floats""" 
    result = pylab.array(vals) 
    mean = sum(result)/len(result) 
    result = result - mean 
    return result/stdDev(result) 
def iScaleFeatures(vals): 
    """Assumes vals is a sequence of floats""" 
    minVal, maxVal = min(vals), max(vals) 
    fit = pylab.polyfit([minVal, maxVal], [0, 1], 1) 
    return pylab.polyval(fit, vals) 



def buildpassengerExamplesSC(fileName,scale): 
    
    data  = getData2(fileName)
    #data['c1']=scale(data['c1'])
    #data['c2']=scale(data['c2'])
    #data['c3']=scale(data['c3'])
    data['age']=scale(data['age'])
    #data['gender']=scale(data['gender'])
    
    
    examples = [] 
    for i in range(len(data['survived'])):
        a = Passenger2(data['c1'][i],data['c2'][i], data['c3'][i],  data['age'][i],data['gender'][i],data['survived'][i])
        examples.append(a)           
    return examples



x_sc=buildpassengerExamplesSC('TitanicPassengers.txt',zScaleFeatures)

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*201 
for i in range (1000):        
    trainset , testset = divide80_20_1000(x_sc) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 601, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

    

print("__________________part 4_________________")
print("Logistic Regressionwith zScaling:") 
print("Averages for all examples 1000 trials with k=0.5")    
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,20,"Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")

pylab.show()
makeHist2(MaxK, 20, "Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 601, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

x_sc=buildpassengerExamplesSC('TitanicPassengers.txt',iScaleFeatures)

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*201 
for i in range (1000):        
    trainset , testset = divide80_20_1000(x_sc) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 601, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)

        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

    

#print("__________________i-scaling_________________")
print("Logistic Regression with iScaling:") 
print("Averages for all examples 1000 trials with k=0.5")    
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,20,"Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 20, "Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 601, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

#________________part 5_________________

#m,f: I has seperated male and female in the first part 

#---------------male-----------------------
print('___________part 5___________')
Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(m) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)
    Ppv_j =[x for x in Ppv if math.isnan(x) == False]
    

print('Logistic Regression with Male and Female Separated:') 
print('Averages for Male Examples 1000 trials with k=0.5')   
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/len(Sens), 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"MALE:Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 25, "MALE:Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('MALE:Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

#-------------------female--------------------
Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(f) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

    

print('Averages for Female Examples 1000 trials with k=0.5')       
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"Female:Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 25, "Female:Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Female:Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

#----------------zscaling-----------------

def sepratepassengerExamples(fileName,scale): 
    data  = getData2(fileName)
    data['age'] = scale(data['age'])
    examples_F = []
    examples_M = []
    for i in range(len(data['survived'])):
        a = Passenger2(data['c1'][i],data['c2'][i], data['c3'][i],  data['age'][i],data['gender'][i],data['survived'][i])
        if a.getgender() == 1:
            examples_M.append(a) 
        else:
            examples_F.append(a)
    return examples_M,examples_F
m_z,f_z=sepratepassengerExamples('TitanicPassengers.txt',zScaleFeatures )

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(m_z) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)
    
print('Logistic Regression with Male and Female Separated with z-scaling:')
print('Averages for Male Examples 1000 trials with k=0.5')    
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"Male:Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 25, "Male:Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Male:Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()
 

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(m_z) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

print("Averages for Female Examples 1000 trials with k=0.5")   
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"Female:Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 25,"Female:Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Female:Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()

#----------------i scaling-----------------

m_i,f_i=sepratepassengerExamples('TitanicPassengers.txt',iScaleFeatures )

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(m_z) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

print('Logistic Regression with Male and Female Separated with i-scaling:')
print('Averages for Male Examples 1000 trials with k=0.5')    
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist(MaxK, 25, "Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()
 

Accr,Sens,Spec,Ppv = [],[],[],[]

c1_w,c2_w,c3_w,age_w,male_w = [],[],[],[],[]

roc = []
MaxK = []
MaxA = []
mean_accur = [None]*251 
for i in range (1000):        
    trainset , testset = divide80_20_1000(m_z) 
    featureVecs, labels = [], []  
    
    for e in trainset:
        featureVecs.append([e.getc1(),e.getc2(),e.getc3(), e.getage() ,e.getgender()])
        labels.append(e.getlabel())  
    model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000).fit(featureVecs, labels)
    c1_w.append(model.coef_[0][0])
    c2_w.append(model.coef_[0][1])
    c3_w.append(model.coef_[0][2])
    age_w.append(model.coef_[0][3])
    male_w.append(model.coef_[0][4])
    ROC=buildROC(model, testset,1, '', plot =False)
    roc.append(ROC)
    
   
    
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset, 1, 0.5)
    accur,sens,spec,ppv=getStats(truePos, falsePos, trueNeg, falseNeg)
    Accr.append(accur)
    Sens.append(sens)
    Spec.append(spec)
    Ppv.append(ppv)
    Ppv_j =[x for x in Ppv if math.isnan(x) == False]###########
   
        
    
    allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
    count=0
    maxAccuracy=0
    maxcount=0
    maxk=0
    for k in range(400, 651, 1):
        truePos, falsePos, trueNeg, falseNeg = applyModel(model, testset,1, k/1000)

        alltruePos.append(truePos)
        allfalsePos.append(falsePos)
        alltrueNeg.append(trueNeg)
        allfalseNeg.append(falseNeg)
        accur=accuracy(truePos, falsePos, trueNeg, falseNeg)
        if mean_accur[k-400] == None:
            mean_accur[k-400]=accur/1000
        else:
            mean_accur[k-400]+=accur/1000
        
   
        allAccuracy.append(accur)
        if maxAccuracy < accur:
            maxAccuracy = accur
            maxcount=count
            maxk=k/1000    
        count+=1
    MAX=max(mean_accur)
    indexk =(mean_accur.index(MAX)+400)/1000
    MaxK.append(maxk)
    MaxA.append(maxAccuracy)

print("Averages for Female Examples 1000 trials with k=0.5")   
print("Mean weight of C1 =",round((sum(c1_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c1_w),3))
print("Mean weight of C2 =",round((sum(c2_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c2_w),3))
print("Mean weight of C3 =",round((sum(c3_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(c3_w),3))
print("Mean weight of Age =",round((sum(age_w)/1000),3),",95% confidence interval =",round(1.96*stdDev(age_w),3))
print("Mean weight of male gender =",round(sum(male_w)/1000,3),",95% confidence interval =",round(1.96*stdDev(male_w),3))
print('Accuracy =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Sensitivity =', round(sum(Sens)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Sens),3)) 
print('pecificity =', round(sum(Accr)/1000, 3),",95% confidence interval =",round(1.96*stdDev(Accr),3)) 
print('Pos. Pred. Val. =', round(sum(Sens)/len(Sens), 3),",95% confidence interval =",round(1.96*stdDev(Sens),3))    
print('Mean AUROC =',round(sum(roc)/1000,3),",95% confidence interval =",round(1.96*stdDev(roc),3))
print("")

makeHist(MaxA,25,"Maxmum Accuracies","Maximum Accuracies","Numbers of Maximum")
pylab.show()
makeHist2(MaxK, 25,"Threshold values k for Maxmum Accuracies", "Thersholds Values k", " Numbers of ks")
pylab.show()
kValues=[k/1000 for k in range(400, 651, 1)]
pylab.plot(kValues,mean_accur)
pylab.plot(indexk, max(mean_accur),'ro')
pylab.annotate((indexk, round(max(mean_accur),3)), xy=(indexk,MAX))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()





#-------------------KNN--------------------
#_________________part 6___________________

print("_____________part 6_____________")
def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.featureDist(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = example.featureDist(e)
        if dist < maxDist:
            #replace farther neighbor by this one
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances

def findKNearestCS(example, exampleSet, k):
    kNearest, similarities = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        similarities.append(example.cosine_similarity(exampleSet[i]))
    maxSim = max(similarities) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        sim = example.cosine_similarity(e)
        if sim < maxSim:
            #replace farther neighbor by this one
            maxIndex = similarities.index(maxSim)
            kNearest[maxIndex] = e
            similarities[maxIndex] = sim
            maxSim = max(similarities)      
    return kNearest, similarities


def KNearestClassify(training, testSet, label, k):
    """Assumes training and testSet lists of examples, k an int
       Uses a k-nearest neighbor classifier to predict
         whether each example in testSet has the given label
       Returns number of true positives, false positives,
          true negatives, and false negatives"""
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for e in testSet:
#        nearest, distances = findKNearest(e, training, k)
        nearest, similarities = findKNearest(e, training, k)
        #conduct vote
        numMatch = 0
        for i in range(len(nearest)):
            if nearest[i].getlabel() == label:
                numMatch += 1
        if numMatch > k//2: #guess label
            if e.getlabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else: #guess not label
            if e.getlabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg


def confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k):
    #print('\nk = ', k)
    print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
    print('                     ', 'TP', '\t', 'FP')
    print('Confusion Matrix is: ', truePos, '\t', falsePos)
    print('                     ', trueNeg, '\t', falseNeg)
    print('                     ', 'TN', '\t', 'FN' )    
    getStats(truePos, falsePos, trueNeg, falseNeg,True)
    return

def findK(training, minK, maxK, numFolds, label): 
    #Find average accuracy for range of odd values of k 
    accuracies = [] 
    for k in range(minK, maxK + 1, 2): 
        score = 0.0 
        for i in range(numFolds): #downsample to reduce computation time
            fold = random.sample(training, min(5000, len(training))) 
            examples, testSet = divide80_20_1000(fold) 
            truePos, falsePos, trueNeg, falseNeg = KNearestClassify(examples, testSet, label, k) 
            score += accuracy(truePos, falsePos, trueNeg, falseNeg) 
            #confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k)
        accuracies.append(score/numFolds)
    k_max=2*accuracies.index(max(accuracies))+1
    
    #for k in range(minK, maxK + 1, 2):
        
        
    return accuracies,k_max


trainset , testset = divide80_20_1000(x) 
truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainset, testset, 1, 3)
print('k-NN Prediction for Survive with k=3:')
confusionMatrix(truePos, falsePos, trueNeg, falseNeg, 3)
#accuracy(truePos, falsePos, trueNeg, falseNeg)
Accurs,max_k=findK(x, 1, 25, 10, 1)
print("K for Maximum Accuracy is:", max_k)
confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k)
true_acc = []
for i in range(1,26,2):
    truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainset, testset, 1, i)
    true_acc.append(accuracy(truePos, falsePos, trueNeg, falseNeg))
   


x_diff =[1,3,5,7,9,11,13,15,17,19,21,23,25]        
pylab.plot(x_diff,true_acc,label ='Real prediction',color ='orange')
pylab.plot(x_diff,Accurs,label = 'n-ford cross validation ')
pylab.xlabel('k values for KNN Regression')
pylab.ylabel("Accuracy")
pylab.legend()
pylab.show()

print("___________part 7___________")
trainsetm , testsetm = divide80_20_1000(m)
trainsetf , testsetf = divide80_20_1000(f)
trainset , testset = divide80_20_1000(x)
truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainsetm, testsetm, 1, 3)
print("For Male:")
confusionMatrix(truePos, falsePos, trueNeg, falseNeg, 3)


truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainsetf, testsetf, 1, 3)
print("For Female:")
confusionMatrix(truePos, falsePos, trueNeg, falseNeg, 3)


truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainset, testset, 1, 3)
print("Combined Predictions Statistics:")
confusionMatrix(truePos, falsePos, trueNeg, falseNeg, 3)



