"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""
from TensorFlowSOM import SOM
import scipy.spatial as spatial
import numpy as np
############

#TODO
 
class SelfOrganizingMap():

    def __init__(self, dict_config):
        
        self.dConfig = dict_config        
        self.prototypeVectors = None
        print ("initialising SOM model")
        
        return None



    ############################################################
    def train(self, Xactual, Yactual):

        # Initialise an SOM object, and train the neural netwrok ..
        self.som = SOM(self.dConfig['iSquaredMapDim'], len(Xactual[0]), self.dConfig['learning_rate'], self.dConfig['max_iter_SOM'])
        print (" - start trining SOM model...")
        self.som.train(Xactual)
        print (" - training SOM completed.")
        
        
        
        # Mapping all the winner neurons to the input instances
        # We overwrite the 1st column to hold the mapping index
        for iIndex in range(len(Xactual)):
            Xactual[iIndex, 0] = self.getIndexOfWinnerNeuron(Xactual[iIndex])     
        print (" - mapping BMU to training data completed.")
        
        
        
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

        # normalising distance rates
        distanceRates = np.array(distanceRates)        
        distanceRates = distanceRates / distanceRates.sum()
        
        setFactors = []
        for iIndex in range(len(Xactual)):
            distanceIndex = self.indexOfList(Xactual[iIndex, 0], mappedNeurons)
            if distanceIndex>=0:                
                setFactors.append(distanceRates[distanceIndex] * Yactual[iIndex])
        
        # calculate weighted avrages 
        setFactors = np.array(setFactors)
        setFactors = np.sum(setFactors, axis=0)               
        return setFactors
            
    
    def indexOfList(self, element, list_element):
        try:
            index_element = list_element.index(element)
            return index_element
        except ValueError:
            return -1   
    
    
    
    def getPredictions(self, Xactual, Yactual, printAccuracy=False):
        
        FactorsPred =[]
        for iIndex in range(len(Xactual)):
            iPV = self.getIndexOfWinnerNeuron(Xactual[iIndex])  
            FactorsPred.append(self.prototypeVectors[iPV])


        if printAccuracy:
            self.doPrintAccuracy(Yactual, FactorsPred)

        return np.array(FactorsPred)
        



    def doPrintAccuracy(self, Yactual, FactorsPred):
        
        '''
        TODO
        here we need to compare the Yactual against the self.outPrototypeVectors
        '''
        
        return





    
    
    




