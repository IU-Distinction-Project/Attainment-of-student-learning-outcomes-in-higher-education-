"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""


###################################################################
from HybridRegression import HybridRegression
from SelfOrganizingMaps import SelfOrganizingMap
import pandas as pd
import numpy as np


class LearningStudentOutcomes():
    
    def __init__(self):
        
        # configuration and setting up
        ###################################################################        
        self.dict_config = {}
        self.dict_config['seed'] = 0
        self.dict_config['max_iter'] = 1000000
        self.dict_config['Cross_validation'] = 5
        self.dict_config['theta3']=.25
        self.dict_config['iSquaredMapDim']= 4
        self.dict_config['learning_rate']=.15
        self.dict_config['max_iter_SOM']=100
        self.dict_config['radius'] = 1.4


        self.dict_config['strPathOfRegPred'] = "../resources/HybridRegressionPred.csv"
        self.pathMainDataSet = "../resources/dumpMainDataset.csv"
        self.pathGMDataset = "../resources/dumpGMDataset.csv"
        self.pathGMUserCourseIDs = "../resources/dumpGMUserCourseIDs.csv"
        self.pathExport_output = "../resources/export_output.csv"


        # dataset configuration
        self.dataSet=None
        self.iDSTestingPosition = 20
        self.iGradeColumnNumber = 18        
        return




    # loading Dataset : training & prediction parts
    ###################################################################
    def loadDataset(self, is_it_for_hybrid_regression=True, is_it_for_training=True):
        

            # We fetch the whole dataset from the local csv file only once
            if self.dataSet is None:                
                self.dataSet = pd.read_csv(self.pathMainDataSet, header=None)
            
            '''
            The main dataset in the file "MainDataset.csv" must contain in the following columns (in CSV format):
                
            0 StudentID	
            1 CourseID
            ============================================
            2    X[0]  CourseTeachingRate	
            3    X[1]  S_completedCourses	
            4    X[2]  S_levelRatio	
            5    X[3]  courseLevelRatio
            6    X[4]  GPA
            7    X[5]  GPAChangeRate
            8    X[6]  TG_L_Last
            9    X[7]  AG_L_Last
            10   X[8]  LG_L_Last
            11   X[9]  TG_L_All
            12   X[10] AG_L_All
            13   X[11] LG_L_All
            14   X[12] TG_Last
            15   X[13] AG_Last
            16   X[14] LG_Last
            17   X[15] TG_All
            18   X[16] AG_All
            19   X[17] LG_All
            ..   X[..] other student features
            ..   X[..] other Course features
            ..   X[..] Grade
            ============================================
            ..   Y[..] Factors 
            ============================================
            '''
                
            if is_it_for_hybrid_regression:

                # This is for Hybrid regression model   
                if is_it_for_training:
                    X = pd.DataFrame(self.dataSet, columns= range(2,18)).values[0:self.iDSTestingPosition]
                    Y = pd.DataFrame(self.dataSet, columns= [18]).values[0:self.iDSTestingPosition]
                    return X, Y
                else:
                    X = pd.DataFrame(self.dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).values[self.iDSTestingPosition:]
                    Y = pd.DataFrame(self.dataSet, columns= [18]).values[self.iDSTestingPosition:]
                    return X, Y
            
            # This is for Multi-label classification
            else:
                if is_it_for_training:
                    X = pd.DataFrame(self.dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).values[0:self.iDSTestingPosition]
                    Y = pd.DataFrame(self.dataSet, columns= [19,20,21,22,23,24,25,26,27,28]).values[0:self.iDSTestingPosition]
                    return X, Y                 
                else:
                    X = pd.DataFrame(self.dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).values[self.iDSTestingPosition:]
                    Y = pd.DataFrame(self.dataSet, columns= [19,20,21,22,23,24,25,26,27,28]).values[self.iDSTestingPosition:]
                    return X, Y 


    def getColumnIndexes(self, iFrom, iTo):        
        return int(",".join(str(iCol) for iCol in np.concatenate([range(iFrom, iTo)])))


    # The big GM matrix that needs to be factorised
    ###################################################################
    def loadBigMatrixGM(self):        
        
        # singular value decomposition (SVD) has been used
        GM = pd.DataFrame(pd.read_csv(self.pathGMDataset, header=None)).values
        
        # mapped student ids with course ids
        SCIDs = pd.DataFrame(pd.read_csv(self.pathGMUserCourseIDs, header=None)).values
        return GM, SCIDs
    
    
    

    # Hybrid regression model for predicting student grades
    ###################################################################
    def performHybridRegression(self):

        hReg = HybridRegression(self.dict_config)
        
        # Training the collaborative filtering model using matrix factorization (GM)
        GM, SCIDs = self.loadBigMatrixGM()
        hReg.trainAndPredictCollaborativeFilteringModel(GM, SCIDs)
        GM=None
        
        
        
        # Loading the training dataset, and train both Lasso regression model and Fuzzy rules ..
        X, Y = self.loadDataset()
        if len(SCIDs) != (len(self.dataSet)-len(Y)):
            raise ValueError("The number of instances used to make predictions must be equal to the size of the mapping list between students and courses in the files ({})".format(self.pathGMUserCourseIDs))        
        SCIDs=None
        hReg.trainLassoModel(X, Y)



        # Loading the testing dataset, and getting the predictions from the three predictors, including the fuzzy rules
        X,Y = self.loadDataset(True, False)
        YgradePred = hReg.getPredictions(X, Y, True)
        X=None
        Y=None

        # Updating the data-frame by adding the predicted grades G
        for i in range(self.iDSTestingPosition, len(self.dataSet)):
            self.dataSet.iat[i, self.iGradeColumnNumber] = YgradePred[i-self.iDSTestingPosition]
        
        



    # Multi-label classification for Predicting Key Factors
    ###################################################################
    def performMultiLabelClassification(self):
        
        # Loading the training dataset that includes [G], and train the self organizing map model
        X,Y = self.loadDataset(False, True)
        SOM = SelfOrganizingMap(self.dict_config)
        SOM.train(X, Y, True)
        
        
        # Loading the testing dataset, and collecting the predictions as prototype vectors 
        X,Y = self.loadDataset(False, False)
        YfactorsPred = SOM.getPredictions(X, Y, True)
        X=None
        Y=None
        
        
        # updating the data-frame by the predicted factors 
        for i in range(self.iDSTestingPosition, len(self.dataSet)):    
            for j in range (self.iGradeColumnNumber+1, len(self.dataSet[0])-1):        
                self.dataSet.iat[i, j]= YfactorsPred[i-self.iDSTestingPosition, j-self.iGradeColumnNumber-1]

        
        
    
    # save all predications from the data-frame to CSV
    ###################################################################
    def saveToCSV(self):
        self.dataSet.to_csv (self.pathExport_output, index = False, header=False)
        print (" - an exported csv file saved ..")
        

    # Print the current data-frame
    ###################################################################
    def printDataFrame(self):
        if self.dataSet is None: self.loadDataset()
        print (self.dataSet)


def main():

    lso = LearningStudentOutcomes()
    lso.performHybridRegression()
    lso.performMultiLabelClassification()
    lso.saveToCSV()    
    print ("Process completed.")
    
    
if __name__ == "__main__":
    main()    

