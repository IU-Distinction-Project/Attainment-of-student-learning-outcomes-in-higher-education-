"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from TensorFlowSOM import SOM
import scipy.spatial as spatial
import numpy as np
import math
############


 
class SelfOrganizingMap():

    def __init__(self, dict_config):
        
        self.dConfig = dict_config        
        self.prototypeVectors = None        
        self.prototypeVectorsSOM_MLL_17 = None
        self.prototypeVectorsML_SOM_19 = None
        print (" - initialising SOM model")
        
        return None



    ############################################################
    def train(self, Xactual, Yactual, computeOtherPrototypeVectors=False):

        # Initialise an SOM object, and train the neural netwrok ..
        self.som = SOM(self.dConfig['iSquaredMapDim'], len(Xactual[0]), self.dConfig['learning_rate'], self.dConfig['max_iter_SOM'])
        print (" - start training SOM model...")
        self.som.train(Xactual)
        print (" - training SOM completed.")
        
        
        
        
        
        
        # Mapping all the winner neurons to the input instances
        # We overwrite the 1st column to hold the mapping index
        # Defining map Distribution 
        #mapDistribution = np.zeros((self.dConfig['iSquaredMapDim'], self.dConfig['iSquaredMapDim']))
        for iIndex in range(len(Xactual)):
            NeuronDim = self.getIndexOfWinnerNeuron(Xactual[iIndex])
            Xactual[iIndex, 0] = NeuronDim
        #    mapDistribution[
        #            self.som.NeuronLocation[NeuronDim][0], 
        #            self.som.NeuronLocation[NeuronDim][1]]+=1
            
        print (" - mapping BMU to training data completed.")
        #print (" - map distribution ...")
        #for i in range(len(mapDistribution)):            
        #    for j in range(len(mapDistribution[0])):
        #        print ("{}\t".format(mapDistribution[i,j]), end="")
        #    print("")
        
        
        
        
        # Get the list of neighbour neurons
        ckdTree = spatial.cKDTree(self.som.NeuronLocation)
        radius = float(self.dConfig['radius'])
        
        
        
        self.prototypeVectors =[]
        for iIndexNeuron in range(len(self.som.NeuronLocation)):
            
            listOfNeighbour = ckdTree.query_ball_point(self.som.NeuronLocation[iIndexNeuron], radius)

            distanceRates =[]
            for innerIndex in range (len(listOfNeighbour)):
                distanceRates.append(self.getGaussianDistanceRate(iIndexNeuron, listOfNeighbour[innerIndex], radius))
                
            self.prototypeVectors.append(
                    self.collectYpartFromTrainingInstances(
                            Xactual, Yactual, listOfNeighbour, distanceRates))
        
        
        print (" - computing [Prototype Vectors] completed.")
        
        
        if computeOtherPrototypeVectors:           
            self.computeMMLPrototypeVectors(Xactual, Yactual, ckdTree, radius)
        
        return None
    


    
    ############################################################
    def getIndexOfWinnerNeuron(self, Xinput):          
        return min([i for i in range(len(self.som.NeuronWeights))],
                            key=lambda x: np.linalg.norm(Xinput - self.som.NeuronWeights[x])) 
    
    


    # Euclidean and Gaussian distances       
    ############################################################
    def getGaussianDistanceRate (self, WinNeuron, NeighborNeuron, radius):        
        dEuclidean = np.linalg.norm(
                self.som.NeuronWeights[WinNeuron] - self.som.NeuronWeights[NeighborNeuron])        
        return np.exp(-np.power(dEuclidean, 2) / (2 * np.power(radius, 2)))
    
    
    
    
    # Collect the training instances with the mapped neurons
    ############################################################
    def collectYpartFromTrainingInstances(self, Xactual, Yactual, mappedNeurons, distanceRates):

        setFactors = []
        duplicatedDistanceRates =[]
        
        for iIndex in range(len(Xactual)):
            distanceIndex = self.indexOfList(Xactual[iIndex, 0], mappedNeurons)
            if distanceIndex>=0:
                duplicatedDistanceRates.append(distanceRates[distanceIndex])
                setFactors.append(Yactual[iIndex])
        
              
        # normalising (duplicated) distance rates          
        duplicatedDistanceRates = np.array(duplicatedDistanceRates)   
        if duplicatedDistanceRates.sum() > 0:
            duplicatedDistanceRates = duplicatedDistanceRates / duplicatedDistanceRates.sum()
        else:
            duplicatedDistanceRates = duplicatedDistanceRates + 1.0
            duplicatedDistanceRates = duplicatedDistanceRates / duplicatedDistanceRates.sum()

            
       
        
        # calculate weighted averages 
        for iIndex in range(len(duplicatedDistanceRates)):
            setFactors[iIndex] = setFactors[iIndex] * duplicatedDistanceRates[iIndex]

        return np.sum(np.array(setFactors), axis=0) 
            
    
    
    
    ############################################################
    def computeMMLPrototypeVectors(self, Xactual, Yactual, ckdTree, radius):
                
        self.prototypeVectorsSOM_MLL_17 = []        
        self.prototypeVectorsML_SOM_19 = []
        
        for iIndexNeuron in range(len(self.som.NeuronLocation)):
            
            # for 2017, mappedNeurons are only the winners
            mappedNeurons = []
            mappedNeurons.append(iIndexNeuron)            
            self.prototypeVectorsSOM_MLL_17.append(
                    self.collectYpartFromTrainingInstancesMML(
                            Xactual, Yactual, mappedNeurons))
            
            # for 2019, mappedNeurons are both the winners plus neighbours 
            mappedNeurons = ckdTree.query_ball_point(self.som.NeuronLocation[iIndexNeuron], radius)
            self.prototypeVectorsML_SOM_19.append(
                    self.collectYpartFromTrainingInstancesMML(
                            Xactual, Yactual, mappedNeurons))
            
        print (" - computing [Prototype Vectors for SOM_MLL_17 and ML_SOM_19] completed.")
                       
            


    # this is to compare our results with the state of the art ...
    ############################################################
    def collectYpartFromTrainingInstancesMML(self, Xactual, Yactual, mappedNeurons):        
        setFactors = []
        for iIndex in range(len(Xactual)):
            if self.indexOfList(Xactual[iIndex, 0], mappedNeurons)>=0:                
                setFactors.append(Yactual[iIndex])
        
        if len(setFactors)==0:
            defaultOutput = [0] * len(Yactual[0])
            setFactors = np.array(defaultOutput)                
        else:
            # calculate a standard average
            setFactors= np.mean(np.array(setFactors), axis=0)
                        
        return setFactors

    
    
    
    
    ############################################################
    def indexOfList(self, element, list_element):
        try:
            index_element = list_element.index(element)
            return index_element
        except ValueError:
            return -1   
    
    
    
    
    ############################################################
    def getPredictions(self, Xactual, Yactual, printAccuracy=False):
                
        MappedNeurons = []
        for iIndex in range(len(Xactual)):
            iPV = self.getIndexOfWinnerNeuron(Xactual[iIndex])  
            MappedNeurons.append(iPV)
            
        Yactual = np.array(Yactual)
                
        
        if printAccuracy:
            
            # SOM_MLL_17         
            FactorsPred =[]
            for iPV in range(len(MappedNeurons)):
                FactorsPred.append(self.prototypeVectorsSOM_MLL_17[MappedNeurons[iPV]])
            SOM_MLL_17 = self.doPrintAccuracy(Yactual, np.array(FactorsPred), " - RMSE[SOM_MLL_17]: {}")
            
            
            # ML_SOM_19
            FactorsPred =[]        
            for iPV in range(len(MappedNeurons)):
                FactorsPred.append(self.prototypeVectorsML_SOM_19[MappedNeurons[iPV]])
            ML_SOM_19 = self.doPrintAccuracy(Yactual, np.array(FactorsPred), " - RMSE[ML_SOM_19]: {}")
        
        
         # Our SOM approach
        FactorsPred =[]
        for iPV in range(len(MappedNeurons)):
            FactorsPred.append(self.prototypeVectors[MappedNeurons[iPV]])
        FactorsPred = np.array(FactorsPred)
                
        
        if printAccuracy:
            Our_SOM = self.doPrintAccuracy(Yactual, FactorsPred, " - RMSE[Our-SOM]: {}")
            print (" - SOM_Epochs: {}".format(self.dConfig["max_iter_SOM"]))

        return FactorsPred, "SOM_MLL_17:\t{}\tML_SOM_19:\t{}\tOur_SOM:\t{}".format(SOM_MLL_17, ML_SOM_19, Our_SOM)
        



    ############################################################
    def doPrintAccuracy(self, Yactual, FactorsPred, msg):     
        if len(Yactual) != len(FactorsPred):
            raise ValueError("Yactual and FactorsPred are not the same!") 
        
        rmse = np.sqrt(
                np.mean(
                    np.power(Yactual - FactorsPred, 2)))
        
        print (msg.format(rmse))
        return rmse



    
    




