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
        print (" - training [Collaborative Filtering Model]")
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
       print (" - training [Lasso Model]")
       self.lassoReg.fit(Xtrain, Ytrain.ravel())
       scores = self.lassoReg.score(Xtrain,Ytrain)
       score_mean = scores.mean()        
       print (" - lassoReg ... \n\t Score_mean ={} \n\t Intercept ={} \n\t Coefficient = {}...".format(score_mean, self.lassoReg.intercept_, self.lassoReg.coef_[0]))        
       return

	
	
	############################################################
    def getPredictions(self, Xpred, Yactual, printToCSV=False):
        
        print (" - getting the hybrid predictions ... ")
        
        # getting specific data from the dataset 
        theta1, theta2 = self.getTheta1And2(Xpred[:,0:1], Xpred[:,1:2])
        studentLevelRatio = Xpred[:,2:3]
        courseLevelRatio = Xpred[:,3:4]

        
        fRules = FuzzyRules()
        
        # get predictions lists
        self.YpredLasso = self.roundIt(self.lassoReg.predict(Xpred))
        YpredFuzzyRules = fRules.getFuzzyRulesPredictions(Xpred)
        self.YpredMatrixFactorization = self.roundIt(self.YpredMatrixFactorization)
                

        # combining the three predictions
        HybridYpred = np.multiply(self.YpredMatrixFactorization, theta1.T)         
        HybridYpred = HybridYpred + np.multiply(YpredFuzzyRules, theta2.T) 
        HybridYpred = HybridYpred + np.multiply(self.YpredLasso, self.dConfig['theta3'])
                        
        HybridYpred = self.roundIt(HybridYpred.T)        
        
        if printToCSV:
            # print also the level
            print (" - saving the hybrid predictions to a CSV file ... ")
            with open(self.dConfig['strPathOfRegPred'],'w') as preCSV:
                preCSV.write("ActualOutput,studentLevelRatio,courseLevelRatio,HybridYpred,YpredMatrixFactorization,YpredFuzzyRules,YpredLasso,theta1,theta2,theta3\n")
                for i in range(len(self.YpredLasso)):
                    preCSV.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                            Yactual[i,0],
                            studentLevelRatio[i,0],
                            courseLevelRatio[i,0],                            
                            HybridYpred[i,0], 
                            self.YpredMatrixFactorization[i], 
                            YpredFuzzyRules[i], 
                            self.YpredLasso[i], 
                            theta1[i,0], 
                            theta2[i,0], 
                            self.dConfig['theta3']))
                                    
		
        
        # print RMSE with Theta3 for only hybrid predictions
        print (" - RMSE for the Hybrid Model= {} and theta3= {}".format(
                self.RMSE(HybridYpred, Yactual), self.dConfig['theta3']))
        
        
        # return the mixed prediction           
        return HybridYpred


    
    def RMSE(self, pred, actual):
        return np.sqrt(((pred - actual) ** 2).mean())


    ############################################################
    def getTheta1And2(self, CourseTeachingRate,	S_completedCourses):        
        theta1 = (CourseTeachingRate * (1- self.dConfig['theta3']))/(CourseTeachingRate+S_completedCourses)
        theta2 = 1-theta1-self.dConfig['theta3']
        return theta1, theta2



    ############################################################
    def roundIt(self, xArray):
        return np.round(xArray, 3)


    