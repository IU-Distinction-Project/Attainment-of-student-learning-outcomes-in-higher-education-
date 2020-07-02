"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from sklearn.decomposition import NMF
from sklearn.linear_model import LassoCV
from FuzzyRules import FuzzyModel
import numpy as np
############################################################
		






class HybridRegression():

    
    ############################################################
    def __init__(self, dict_config):

        self.dConfig = dict_config	        

        self.mf = NMF(
                init='random', 
                random_state=self.dConfig['seed'], 
                beta_loss='frobenius', 
                solver='cd', 
                max_iter=self.dConfig['max_iter'], 
                l1_ratio=.8, 
                alpha=0.1) # alpha=0 means => no regularization 
        self.YpredMatrixFactorization = None
          
        
        self.lassoReg = LassoCV(
                cv=self.dConfig['Cross_validation'], 
                random_state=self.dConfig['seed'], 
                max_iter=self.dConfig['max_iter'])
        self.YpredLasso = None
        
        
        self.FuzzyRules = FuzzyModel(self.dConfig['DGapRatioForAT_AL'])  
        self.YpredFuzzy = None
        
        
                
        
        
        
    ############################################################    
    def trainAndPredictCollaborativeFilteringModel(self, GM, SCIDs): 
        
        # we train the the collaborative filtering model (matrix factorization)
        print (" - training [Collaborative Filtering Model]...")
        S = self.mf.fit_transform(GM)
        C = self.mf.components_
        self.mf = np.dot(S, C)
      
        
        # Filling out the predicted values
        print (" - predicting [Collaborative Filtering Model]...")
        self.YpredMatrixFactorization = np.random.random(len(SCIDs))
        for i in range(len(SCIDs)):
            self.YpredMatrixFactorization[i] = self.mf[SCIDs[i, 0], SCIDs[i, 1]]  
        
        self.mf = None # to free the mem            
        return
    
    
    
    ############################################################
    def trainLassoModel(self, Xtrain, Ytrain):         
       print (" - training [Lasso Model]")
       self.lassoReg.fit(Xtrain, Ytrain.ravel())
       
       print (" - lassoReg ... \n\t Score_mean ={} \n\t Intercept ={} \n\t Coefficient = {}...".format(
               self.lassoReg.score(Xtrain,Ytrain).mean(), 
               self.lassoReg.intercept_, 
               self.lassoReg.coef_[0]))        
       return

	
    
    ############################################################
    def trainFuzzyModel(self, Xtrain):
        print (" - training [Fuzzy Model]...")
        
        for i in range(len(Xtrain)):
            
            StudentID = Xtrain[i,0]
            CourseID = Xtrain[i,1]            
            StudentLevel = Xtrain[i,2]
            CourseLevel = Xtrain[i,3]            
            CourseCategory = Xtrain[i,4]
              
            self.FuzzyRules.train(
                    StudentID, 
                    StudentLevel,
                    CourseID, 
                    CourseLevel, 
                    CourseCategory)        


    ############################################################
    def predictFuzzyModel(self, Xpred, Yactual):
        print (" - grade predictions [Fuzzy Model]...")     
        self.YpredFuzzy = self.FuzzyRules.getFuzzyPredictions(Xpred, Yactual)
        self.FuzzyRules = None # to free the mem
        return self.YpredFuzzy
    
    
    
	
	############################################################
    def getPredictions(self, Xpred, Yactual, printToCSV=False):
        
        print (" - getting the hybrid predictions ... ")
        
        # getting details from X        
        courseTeachingRate = Xpred[:,1:2]
        StudentCompletedCourses = Xpred[:,8:9]
        
        
        theta1, theta2 = self.getTheta1And2(
                courseTeachingRate, 
                StudentCompletedCourses)
        
                
        # get predictions lists
        self.YpredLasso = self.roundIt(self.lassoReg.predict(Xpred))
        self.YpredMatrixFactorization = self.roundIt(self.YpredMatrixFactorization)
                
        

        # combining the three predictions
        HybridYpred = np.multiply(self.YpredMatrixFactorization, theta1.T)         
        HybridYpred = HybridYpred + np.multiply(self.YpredFuzzy, theta2.T) 
        HybridYpred = HybridYpred + np.multiply(self.YpredLasso, self.dConfig['theta3'])
        HybridYpred = self.roundIt(HybridYpred.T)        

        
        if printToCSV:

            print (" - saving the hybrid predictions to a CSV file ... ")            
            
            StudentLevelRatio = Xpred[:,9:10]
            courseLevelRatio = Xpred[:,2:3]
                      
            with open(self.dConfig['strPathOfRegPred'],'w') as preCSV:
                preCSV.write("ActualOutput,courseLevelRatio,courseTeachingRate,HybridYpred,YpredMatrixFactorization,YpredFuzzyRules,YpredLasso,theta1,theta2,theta3\n")
                for i in range(len(self.YpredLasso)):
                    preCSV.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            Yactual[i,0],
                            StudentLevelRatio[i,0],
                            courseLevelRatio[i,0],                            
                            HybridYpred[i,0], 
                            self.YpredMatrixFactorization[i], 
                            self.YpredFuzzy[i], 
                            self.YpredLasso[i], 
                            theta1[i,0], 
                            theta2[i,0], 
                            self.dConfig['theta3']))
                                    
		
        
        # print RMSE with Theta3 for only hybrid predictions
        print (" - RMSE for the Hybrid Model= {} and theta3= {}".format(
                self.getRMSE(HybridYpred, Yactual), self.dConfig['theta3']))
        
        
        # return the mixed prediction           
        return HybridYpred




    ############################################################
    def getRMSE(self, pred, actual):
        return np.sqrt(((pred - actual) ** 2).mean())


    ############################################################
    def getTheta1And2(self, courseTeachingRate,	StudentCompletedCourses):        
        theta1 = (courseTeachingRate * (1- self.dConfig['theta3']))/(courseTeachingRate + StudentCompletedCourses)
        theta2 = 1-theta1-self.dConfig['theta3']
        return theta1, theta2


    ############################################################
    def roundIt(self, xArray):
        return np.round(xArray, 3)


    