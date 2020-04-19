"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from sklearn.decomposition import NMF
from sklearn.linear_model import LassoCV
from FuzzyRules import FuzzyRules
import numpy as np
############################################################
		






class HybridRegression():

    
    ############################################################
    def __init__(self, dict_config):

        self.dConfig = dict_config	        

        self.mf = NMF(init='random', random_state=self.dConfig['seed'], beta_loss='frobenius', solver='cd', max_iter=self.dConfig['max_iter'], l1_ratio=.5, alpha=.5) # alpha=0 means => no regularization 
        self.YpredMatrixFactorization = None
                
        self.lassoReg = LassoCV(cv=self.dConfig['Cross_validation'], random_state=self.dConfig['seed'], max_iter=self.dConfig['max_iter'])
        self.YpredLasso = None
        return
        
        
        
        
        
        
    ############################################################    
    def trainAndPredictCollaborativeFilteringModel(self, GM, SCIDs): 
        
        # we train the the collaborative filtering model (matrix factorization)
        print ("Training [Collaborative Filtering Model]")
        S = self.mf.fit_transform(GM)
        C = self.mf.components_
        self.mf = np.dot(S, C)
      
        
        # Filling out the predicted values
        self.YpredMatrixFactorization = np.random.random(len(SCIDs))
        for i in range(len(SCIDs)):
            self.YpredMatrixFactorization[i] = self.mf[SCIDs[i, 0], SCIDs[i, 1]]    
        return
    
    
    
    ############################################################
    def trainLassoModel(self, Xtrain, Ytrain):         
       print ("Training [Lasso Model]")
       self.lassoReg.fit(Xtrain, Ytrain.ravel())
       scores = self.lassoReg.score(Xtrain,Ytrain)
       score_mean = scores.mean()        
       print ("LassoReg ... \n\t Score_mean ={} \n\t Intercept ={} \n\t Coefficient = {}".format(score_mean, self.lassoReg.intercept_, self.lassoReg.coef_))        
       return

	
	
	############################################################
    def getPredictions(self, Xpred, printToCSV=False):
        
        theta1, theta2 = self.getTheta1And2(Xpred[:,0:1], Xpred[:,1:2])
        fRules = FuzzyRules()
        
        # get predictions lists
        self.YpredLasso = self.roundIt(self.lassoReg.predict(Xpred))
        YpredFuzzyRules = fRules.getFuzzyRulesPredictions(Xpred)
        self.YpredMatrixFactorization = self.roundIt(self.YpredMatrixFactorization)
                
        
        HybridYpred = np.multiply(self.YpredMatrixFactorization, theta1.T) + np.multiply(YpredFuzzyRules, theta2.T) + np.multiply(self.YpredLasso, self.dConfig['theta3'])
        HybridYpred = self.roundIt(HybridYpred.T)
          
        
        if printToCSV:
            with open(self.dConfig['strPathOfRegPred'],'w') as preCSV:
                preCSV.write("HybridYpred,YpredMatrixFactorization,YpredFuzzyRules,YpredLasso,theta1, theta2, theta3 \n")
                for i in range(len(self.YpredLasso)):
                    preCSV.write("{},{},{},{},{},{},{}\n".format(HybridYpred[i,0], self.YpredMatrixFactorization[i], YpredFuzzyRules[i], self.YpredLasso[i], theta1[i,0], theta2[i,0], self.dConfig['theta3']))
                                    
		# return the mixed prediction           
        return HybridYpred


    
    
    ############################################################
    def getTheta1And2(self, CourseTeachingRate,	S_completedCourses):        
        theta1 = (CourseTeachingRate * (1- self.dConfig['theta3']))/(CourseTeachingRate+S_completedCourses)
        theta2 = 1-theta1-self.dConfig['theta3']
        return theta1, theta2



    ############################################################
    def roundIt(self, xArray):
        return np.round(xArray, 3)


    