"""
Original version is developed by Nikola Živković (zivkovic.nikola87@gmail.com)
https://gist.github.com/NMZivkovic/3e5a5623de009103febb3a6bef61b140#file-somtf-py

and improved and customised by Abdullah Alshanqiti (a.m.alshanqiti@gmail.com)
on 16-4-2020

"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
 
class SOM(object):
    
    
    def __init__(self, iSquaredMapDim, iInputLength, learning_rate, maxIter):
                
        
        self.iSquaredMapDim = iSquaredMapDim
        iNeurons = self.iSquaredMapDim * self.iSquaredMapDim
        self.maxIter = maxIter
        self.dRadius = self.iSquaredMapDim * .6
        self.tfGraph = tf.Graph()
        

            
        # Initialize a TensorFlow computation graph
        with self.tfGraph.as_default():                  

            
            # Initializing variables
            tf.set_random_seed(0)
            self.NeuronWeights = tf.Variable(tf.random_normal([iNeurons, iInputLength]))
            self.NeuronLocation = self.generateIndexMatrix()            
                        
            
            
            # Input placeholders
            self.inputPlaceholder = tf.placeholder("float", [iInputLength])
            self.iterInputPlaceholder = tf.placeholder("float")
 
    
    
            # Calculating best mapping unit (BMU) and its location
            input_matix = tf.stack([self.inputPlaceholder for i in range(iNeurons)])
            euclidenDistances = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.NeuronWeights, input_matix), 2), 1))
            bmu = tf.argmin(euclidenDistances, 0)
            mask = tf.pad(tf.reshape(bmu, [1]), np.array([[0, 1]]))
            size = tf.cast(tf.constant(np.array([1, 2])), dtype=tf.int64)
            bmu_location = tf.reshape(tf.slice(self.NeuronLocation, mask, size), [2])
     
        
        
            # Calculate learning rate and radius
            decay_function = tf.subtract(1.0, tf.div(self.iterInputPlaceholder, self.maxIter))
            _current_learning_rate = tf.multiply(learning_rate, decay_function)
            _current_radius = tf.multiply(self.dRadius, decay_function)  
                        
    
            # Adapt learning rate to each neuron based on position
            bmu_matrix = tf.stack([bmu_location for i in range(iNeurons)])
            bmu_distance = tf.reduce_sum(tf.pow(tf.subtract(self.NeuronLocation, bmu_matrix), 2), 1)
            
                        
            
            # Gaussian distrbution 
            gaussianNeighbourhood = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance, "float32"), tf.pow(_current_radius, 2))))
            learning_rate_matrix = tf.multiply(_current_learning_rate, gaussianNeighbourhood)
    
    
    
            # Update all the weights
            multiplytiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_matrix, np.array([i]), np.array([1])), [iInputLength])
                                               for i in range(iNeurons)])
            delta = tf.multiply(
                multiplytiplier,
                tf.subtract(tf.stack([self.inputPlaceholder for i in range(iNeurons)]), self.NeuronWeights))                
                         
            new_weights = tf.add(self.NeuronWeights, delta)
            self._training = tf.assign(self.NeuronWeights, new_weights)                                       
 
    
    
            #Initilize session and run it
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())            
            return
 
    
    
    ############################################################
    def generateIndexMatrix(self):
        return tf.constant(np.array(list(self.getIndexes())))
        
    def getIndexes(self):
        for j in range(self.iSquaredMapDim):
            for i in range(self.iSquaredMapDim):
                yield np.array([i, j])
                
               
                
    ############################################################            
    def train(self, inputVectors):
        for iIter in range(self.maxIter):
            print(".", end='')
            if iIter % 100 ==0 and iIter>1:
                print(". ", iIter)
            
            for input_vect in inputVectors:                                
                self._sess.run(self._training,
                               feed_dict={self.inputPlaceholder: input_vect,
                                          self.iterInputPlaceholder: iIter})
                
        print(".")
        self.NeuronWeights = list(self._sess.run(self.NeuronWeights))                
        self.NeuronLocation = list(self._sess.run(self.NeuronLocation))
        return self.NeuronWeights, self.NeuronLocation
            
            
    

    
   
    
    
    