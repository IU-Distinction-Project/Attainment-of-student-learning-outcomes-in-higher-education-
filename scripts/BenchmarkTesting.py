"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

###################################################################
from SelfOrganizingMaps import SelfOrganizingMap
import pandas as pd
import numpy as np


class SOMBenchmarkTesting():
    
    def __init__(self):
        
        # configuration and setting up
        ###################################################################        
        self.dict_config = {}
        self.dict_config['seed'] = 0			# 5,7,88,45,69,59,43,67,27
        self.dict_config['iSquaredMapDim']= 6
        self.dict_config['learning_rate']=.1
        self.dict_config['max_iter_SOM']= 500
        self.dict_config['radius'] = 2.4


        self.defaultPath = "../resources/StateOfTheArtDatasets/{}.csv"
        self.strPathRMSE = "../resources/StateOfTheArtDatasets/rmseForAll.txt"
        self.dataSet=None    
        return




    # loading Dataset : training & prediction parts
    ###################################################################
    def getDatasetslicing(self, ColumnOfFirstLabelAt, iTestingPosition, is_it_for_training=True):
                  
        if is_it_for_training:            
            X = pd.DataFrame(self.dataSet, columns= range(len(self.dataSet.columns))).values[0:iTestingPosition]
            Y = pd.DataFrame(self.dataSet, columns= range(ColumnOfFirstLabelAt, len(self.dataSet.columns))).values[0:iTestingPosition]
            return X, Y
        
        else:
            X = pd.DataFrame(self.dataSet, columns= range(len(self.dataSet.columns))).values[iTestingPosition:]
            Y = pd.DataFrame(self.dataSet, columns= range(ColumnOfFirstLabelAt, len(self.dataSet.columns))).values[iTestingPosition:]
            #set zeros as default values
            for i in range(len(X)):
                for j in range (ColumnOfFirstLabelAt, len(self.dataSet.columns)): 
                    X[i,j]=1.0
                    
            return X, Y 
      
    
    
    
    # Multi-label classification for Predicting Key Factors
    ###################################################################
    def runSOMClassification(self, ds, ColumnOfFirstLabelAt, iTestingPosition, toAppend='a'):
        
        # Fetching the whole dataset ...       
        self.dataSet = pd.read_csv(self.defaultPath.format(ds), header=None)
        
        X, Y = self.getDatasetslicing(ColumnOfFirstLabelAt, iTestingPosition)
        SOM = SelfOrganizingMap(self.dict_config)
        SOM.train(X, Y, True)
        
        X,Y = self.getDatasetslicing(ColumnOfFirstLabelAt, iTestingPosition, False)
        _, RMSEs = SOM.getPredictions(X, Y, True)
    
        
        # print RMSE to file .. here
        with open(self.strPathRMSE, toAppend) as printRMSE:
            printRMSE.write("{}: {}\n".format(ds, RMSEs))
                
        return RMSEs
        
        
        

def main():
    
    somTesting = SOMBenchmarkTesting()        
    somTesting.runSOMClassification("cal500", 68, 370, 'w')
    somTesting.runSOMClassification("birds", 258, 323)
    somTesting.runSOMClassification("emotions", 72, 391)
    somTesting.runSOMClassification("flags", 10, 129)
    somTesting.runSOMClassification("scene", 294, 1211)
    somTesting.runSOMClassification("yeast", 103, 1500)
    
        
    print ("Process completed.")
    
    
if __name__ == "__main__":
    main()

