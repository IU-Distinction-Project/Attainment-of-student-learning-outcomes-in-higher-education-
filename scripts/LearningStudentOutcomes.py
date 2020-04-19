"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

############
from HybridRegression import HybridRegression
from SelfOrganizingMaps import SelfOrganizingMap
import pandas as pd




###################################################################
###################################################################
############ configuration and setting up
dict_config = {}
dict_config['seed'] = 0
dict_config['max_iter'] = 1000000
dict_config['Cross_validation'] = 5
dict_config['theta3']=.25
dict_config['nEpochs'] = 1
dict_config['batchSize'] = 1
dict_config['strPathOfRegPred'] = "../resources/HybridRegressionPred.csv"


pathMainDataSet = "../resources/dumpMainDataset.csv"
pathGMDataset = "../resources/dumpGMDataset.csv"
pathGMUserCourseIDs = "../resources/dumpGMUserCourseIDs.csv"


dataSet=None
iDSTestingPosition = 20
iGradeColumnNumber = 18








###################################################################
###################################################################
############ loading Dataset : training & prediction parts
def loadDataset(is_it_for_hybrid_regression=True, is_it_for_training=True, dataSet=None):

    X = None
    Y = None
    if dataSet is None:
        # We fetch the whole dataset from the local csv file
        # This is done only once
        dataSet = pd.read_csv(pathMainDataSet, header=None)
        
        
    if is_it_for_hybrid_regression:
        
        # This is for Hybrid regression model
        if is_it_for_training:
            X = pd.DataFrame(dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).values[0:iDSTestingPosition]
            Y = pd.DataFrame(dataSet, columns= [18]).values[0:iDSTestingPosition]
            return dataSet, X, Y
        else:
            X = pd.DataFrame(dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]).values[iDSTestingPosition:]
            Y = None #pd.DataFrame(dataSet, columns= [18]).values[iDSTestingPosition:]
            return dataSet, X, Y
    
    else:
        
        # This is for Multi-label classification
        if is_it_for_training:
            X = pd.DataFrame(dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).values[0:iDSTestingPosition]
            Y = pd.DataFrame(dataSet, columns= [19,20,21,22,23,24,25,26,27,28]).values[0:iDSTestingPosition]
            return dataSet, X, Y 
        
        else:
            X = pd.DataFrame(dataSet, columns= [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).values[iDSTestingPosition:]
            Y = None #pd.DataFrame(dataSet, columns= [19,20,21,22,23,24,25,26,27,28]).values[iDSTestingPosition:]
            return dataSet, X, Y 



def loadBigMatrixGM():
    # singular value decomposition (SVD) has been used
    
    # big GM matrix that needs to be factorised
    GM = pd.DataFrame(pd.read_csv(pathGMDataset, header=None)).values
    
    # mappede student ids with course ids
    SCIDs = pd.DataFrame(pd.read_csv(pathGMUserCourseIDs, header=None)).values
    return GM, SCIDs
    
    
    


###################################################################
###################################################################
############ Hybrid regression model for predicting student grades
hReg = HybridRegression(dict_config)

# Training
GM, SCIDs = loadBigMatrixGM()
hReg.trainAndPredictCollaborativeFilteringModel(GM, SCIDs)
GM=None
SCIDs=None


dataSet, X, Y = loadDataset()
hReg.trainLassoModel(X, Y)
X=None
Y=None


# Prediction  .. [it includes the prediction by also fuzzy rules]
dataSet, X,Y = loadDataset(True, False, dataSet)
YgradePred = hReg.getPredictions(X, True)


# Updating the data-frame by adding the predicted grades G
for i in range(iDSTestingPosition, len(dataSet)):
    dataSet.iat[i, iGradeColumnNumber] = YgradePred[i-iDSTestingPosition]

hReg = None
X=None
Y=None
YgradePred = None



'''

###################################################################
###################################################################
############ Multi-label classification for Predicting Key Factors
dataSet, X,Y = loadDataset(False, True, dataSet)
SOM = SelfOrganizingMap(X, Y, dict_config)
#SOM.train()

dataSet, X,Y = loadDataset(False, False, dataSet)
YfactorsPred = SOM.getPredictions(X,Y)

# updating the data-frame by the predicted factors 
for i in range(iDSTestingPosition, len(dataSet)):    
    for j in range (iGradeColumnNumber+1, len(dataSet[0])-1):        
        dataSet.iat[i, j]= YfactorsPred[i-iDSTestingPosition, j-iGradeColumnNumber-1]






###################################################################
###################################################################
############ Saving all predications to *.csv
dataSet.to_csv ('../resources/export_dataSet.csv', index = False, header=False)
print ("Exported csv file saved")


'''

print ("Process completed")
