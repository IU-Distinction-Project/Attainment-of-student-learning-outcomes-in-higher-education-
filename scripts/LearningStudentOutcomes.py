"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

###################################################################
from HybridRegression import HybridRegression
from SelfOrganizingMaps import SelfOrganizingMap
import pandas as pd


class LearningStudentOutcomes():
    
    def __init__(self):
        
        # configuration and setting up
        ###################################################################        
        self.dict_config = {}
        self.dict_config['seed'] = 0	# 5,7,88,45,69,59,43,67,27
        self.dict_config['max_iter'] = 1000000
        self.dict_config['Cross_validation'] = 5
        self.dict_config['theta3']=.35
        self.dict_config['iSquaredMapDim']= 6
        self.dict_config['learning_rate']=.1
        self.dict_config['max_iter_SOM']= 100000
        self.dict_config['radius'] = 4.5
        self.dict_config['DGapRatioForAT_AL']=1.5


        self.dict_config['strPathOfRegPred'] = "../resources/HybridRegressionPred.csv"
        self.pathMainDataSet = "../resources/MainDataset.csv"
        self.pathGMDataset = "../resources/GMDataset.csv"
        self.pathGMUserCourseIDs = "../resources/GMStudentCourseIDs.csv"
        self.pathExport_output = "../resources/export_output.csv"


        # dataset configuration
        self.dataSet=None
        self.iDSTestingPosition = 165360
        self.iGradeColumnNumber = 16        
        return




    # loading Dataset : training & prediction parts
    ###################################################################
    def loadDataset(self, is_it_for_hybrid_regression=True, is_it_for_training=True):
        

            # We fetch the whole dataset from the local csv file only once
            if self.dataSet is None:                
                self.dataSet = pd.read_csv(self.pathMainDataSet, header=None)
            
            '''
            The main dataset in the file "MainDataset.csv" must contain the following columns (in CSV format):
                
            0  StudentID	
            1  CourseID
            2  Student Level
            3  Course Level	
            4  Course Category	
            ============================================
            5  Course Category Ratio	
            6  Course Teaching Rate	
            7  course Level Ratio	
            8  CourseTG	
            9  CourseAA	
            10 CourseAG	
            11 CourseBA	
            12 CourseLG	
            ============================================
            13 Student Completed Courses 
            14 Student Level Ratio
            15 Student average mark
            16 Grade
            ============================================
            other student features
            other Course features    
            ============================================
            Y[..] Factors 
            ============================================
            '''
                
            if is_it_for_hybrid_regression:

                # This is for Hybrid regression model   
                if is_it_for_training:
                    X = pd.DataFrame(self.dataSet, columns= range(5, self.iGradeColumnNumber-1)).values[0:self.iDSTestingPosition]
                    Y = pd.DataFrame(self.dataSet, columns= [self.iGradeColumnNumber]).values[0:self.iDSTestingPosition]
                    return X, Y
                else:
                    X = pd.DataFrame(self.dataSet, columns= range(5, self.iGradeColumnNumber-1)).values[self.iDSTestingPosition:]
                    Y = pd.DataFrame(self.dataSet, columns= [self.iGradeColumnNumber]).values[self.iDSTestingPosition:]
                    return X, Y
            
            # This is for Multi-label classification
            else:
                if is_it_for_training:
                    #self.iGradeColumnNumber+1
                    X = pd.DataFrame(self.dataSet, columns= range(5, len(self.dataSet.columns))).values[0:self.iDSTestingPosition]
                    Y = pd.DataFrame(self.dataSet, columns= range(self.iGradeColumnNumber+1, len(self.dataSet.columns))).values[0:self.iDSTestingPosition]
                    return X, Y                 
                else:
                    X = pd.DataFrame(self.dataSet, columns= range(5, len(self.dataSet.columns))).values[self.iDSTestingPosition:]
                    Y = pd.DataFrame(self.dataSet, columns= range(self.iGradeColumnNumber+1, len(self.dataSet.columns))).values[self.iDSTestingPosition:]
                    #set Zero to pred. values
                    for i in range(len(X)):
                        for j in range (self.iGradeColumnNumber+1-5, len(self.dataSet.columns)-5): 
                            X[i,j]=0.0
                            
                    return X, Y 

    
    # Dataset for fuzzy model
    ###################################################################
    def loadDatasetForFuzzyModel(self, is_it_for_training=True):
        
        if is_it_for_training:
            X = pd.DataFrame(self.dataSet, columns= range(0, 5)).values[0:self.iDSTestingPosition]
            Y = pd.DataFrame(self.dataSet, columns= [self.iGradeColumnNumber]).values[0:self.iDSTestingPosition]
        else:
            X = pd.DataFrame(self.dataSet, columns= range(0, 5)).values[self.iDSTestingPosition:]          
            Y = pd.DataFrame(self.dataSet, columns= [self.iGradeColumnNumber]).values[self.iDSTestingPosition:]
        return X, Y
            


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
            raise ValueError("The number of instances used to make predictions must be equal to the size of the mapping list between students and courses in the files ({}): len(SCIDs)={}, len(self.dataSet)={}, len(Y)={} ".format(self.pathGMUserCourseIDs, len(SCIDs), len(self.dataSet), len(Y)))        
        SCIDs=None
        hReg.trainLassoModel(X, Y)



        # training the fuzzy model
        X,_ = self.loadDatasetForFuzzyModel()
        hReg.trainFuzzyModel(X)
        X, Y = self.loadDatasetForFuzzyModel(False)
        hReg.predictFuzzyModel(X, Y)



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
        YfactorsPred,_ = SOM.getPredictions(X, Y, True)
        X=None
        Y=None
                
        # updating the data-frame by the predicted factors 
        for i in range(self.iDSTestingPosition, len(self.dataSet)):    
            for j in range (self.iGradeColumnNumber+1, len(self.dataSet.columns)):        
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
        
    def printDataSet(self):
        X, Y = self.loadDataset(False, False)
        print (X)
        print (Y)
        

        
        
        
        
        
        

def main():

    lso = LearningStudentOutcomes()     
    lso.performHybridRegression()
    lso.performMultiLabelClassification()
    lso.saveToCSV()
    print ("Process completed.")
    
    
if __name__ == "__main__":
    main()

