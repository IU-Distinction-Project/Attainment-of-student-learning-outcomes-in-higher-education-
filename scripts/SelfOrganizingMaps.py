"""
Created on 16-4-2020

@author: Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
"""

import numpy as np
############

#TODO
 
class SelfOrganizingMap():

    def __init__(self, Xt, Yt, dict_config):
        self.Xtrain = Xt
        self.Ytrain = Yt
        self.dConfig = dict_config		
        self.SOM = None


		
    def train(self):
        return None
    
    
    def getPredictions(self, Xpred, Ypred):
        
        # temp: just for testing
        return np.random.random((len(Ypred), len(Ypred[0])))
        



