"""
This file implements a Hopfield Network model.

Relevant book chapters:
    - http://neuronaldynamics.epfl.ch/online/Ch17.S2.html

"""
######## This code is a modification of the one provided by : 

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

import matplotlib.pyplot as plt
import numpy as np
from copy import copy

import random as rand



class HopfieldNetwork:
    """Implements a Hopfield network of size N.

    Attributes:
        N (int): Number of neurons
        patternsInk (numpy.ndarray): Array of stored ink patterns.
        patternsWord (numpy.ndarray): Array of stored word patterns.
        congru (numpy.ndarray): Array of stored congruent patterns.
        incongru (numpy.ndarray): Array of incongruent pattern used for 
                                    initialization
        config (int): index of word part of incongru in patternsWord
        weight (numpy.ndarray): Array of stored weights.
        x (numpy.ndarray): Network state
    """ 

    def __init__(self, N):
        self.N = N

    def make_pattern(self, P=1, k=0.5, ratio=0.5):
        """Creates and stores Ink and Word patterns to the
        network. And concatenates them in congruent way.
        Finally, the network is trained to learn the congruent patterns 

        Args:
            P (int, optional): number of patterns
                (used only for random patterns)
            k (float, optional): N*k is the length on the ink patterns
            ratio (float, optional): percentage of 'on' pixels
                                    in random patterns, so white pixels

        """
        
        self.patternsInk = -np.ones((P, int(self.N*k)),int)
        idxInk = int(ratio*self.N*k)#number of 'on' pixels 
        self.patternsWord = -np.ones((P, self.N-len(self.patternsInk[0])),int)
        idxWord = int(ratio*self.N*(1-k))#number of 'on' pixels 
        
        for i in range(P):
            self.patternsInk[i, :idxInk] = 1 #we turn 'on' the idx first pixels
                #of each pattern
            self.patternsWord[i, :idxWord] = 1 #we turn 'on' the idx first pixels
                #of each pattern
            self.patternsInk[i] = np.random.permutation(self.patternsInk[i])
                #randomly permutes the 'on' and 'off' pixels in each pattern
            self.patternsWord[i] = np.random.permutation(self.patternsWord[i])
                #randomly permutes the 'on' and 'off' pixels in each pattern

        self.congru=np.concatenate((self.patternsInk,self.patternsWord),axis=1)
                #concatenate i^th ink pattern with i^th word pattern
        self.weight = np.zeros((self.N, self.N))
        
        #we teach the congru patterns to the network
        for i in range(self.N): #for each pixel
            self.weight[i] = 1./self.N * (
                np.sum(
                    self.congru[kk, i] * self.congru[kk]
                    for kk in range(P) #kk runs on the number
                    #of patterns
                )
            )
            self.weight[i,i]=0 #we impose no autaptic connections



    def dynamic(self,update='asynch'):
        """Executes one timestep of the dynamics
        
        Args:update (string, optional): update method 'synch' or 'aysnch'
            
        
        """
        
        
        idx=range(self.N)
        h=np.ones(self.N) 
        
        if update=='asynch':
            
            idx=np.random.permutation(idx)#asychronous update with random order
            for ii in idx:
                h[ii]=np.sum(self.weight*self.x,axis=1)[ii]#we compute the 
                        #influence of the current state on the ii^th pixel 
                if h[ii]==0:
                    self.x[ii]=1;#we define sign(0)=1 and update the current 
                else:   #state with the new value of the pixel 
                    self.x[ii]=np.sign(h[ii])                 

                        
        elif update=='synch':
            h = np.sum(self.weight*self.x, axis=1) #everything is computed at once 
            
            for ii in idx:            
                if h[ii]==0:
                    self.x[ii]=1;
                else:
                    self.x[ii]=np.sign(h[ii])  
        
        else: raise ValueError('Update method unknown, choose between asynch or\
         synch')
                
    def overlap(self, mu, typepattern='congru'):
        """Computes the overlap of the current state with
        pattern number mu.

        Args:
            mu (int): The index of the congruent pattern to
                compare with. mu<P
            typepattern (string, optional): decides if we compare with a congruent
                pattern or the initial incongruent one.
        """

        if typepattern=='congru':
        
            return 1./self.N*np.sum(self.congru[mu]*self.x)
        
        if typepattern=='incongru':
        
            return 1./self.N*np.sum(self.incongru*self.x)    
        
        else:
            raise ValueError(
                'Type of pattern does not exist. Choose congru or incongru'
            )
        

        

    def run(self, t_max=20, mu=0, typepattern='congru', update='asynch'):
        """Runs the dynamics and optionally plots it.

        Args:
            t_max (float, optional): Timesteps to simulate
            mu (int, optional): Pattern number to use
                as initial pattern for the network state (< P)
            typepattern (string, optional ): type of pattern we want to use for
                initialization, congru or incongru
            do_plot (bool, optional): Plot the network as it is
                updated
            update (string, optional): update method 'synch' or 'aysnch'

        Raises:
            IndexError: Raised if given pattern index is too high.
            RuntimeError: Raised if no patterns have been created.
        """
        try:
            self.congru
        except AttributeError:
            raise RuntimeError(
            'No congruent patterns created: please ' +
            'use make_pattern to create at least one pattern.'
            )
        try:
            self.congru[mu]
        except:
            raise IndexError('Pattern index too high (has to be < P)')

        if typepattern=='congru':
            # set the initial state of the net
            self.x = copy(self.congru[mu])# we copy it because we will modify x
                    #but we do not want to modify self.congru[mu]
            
        elif typepattern=='incongru':
            P=len(self.congru)
            idxRandom=rand.randint(0,P-1) #we select a random word color
            
            while idxRandom==mu:
                idxRandom=rand.randint(0,P-1) #but different from the congruent ink color
               
            self.config=idxRandom #we keep track of the used word color
             # set the initial state of the net 
            self.incongru=np.concatenate((self.patternsInk[mu],self.patternsWord[idxRandom]),axis=0)
            self.x=copy(self.incongru)
            
        else :
            raise ValueError(
            'Wrong type of pattern, try congru or incongru' )

        x_old = copy(self.x)

        for i in range(t_max): #while there is still time (or not converged)

            # run a step
            self.dynamic(update=update)            
            i_fin = i
            if np.sum(np.abs(x_old-self.x)) == 0: #if the new state is the same
                break                       #as the previous one we stop
            x_old = copy(self.x)
            
        overlapInk=self.overlap(mu=mu) #compute overlap with congruent color of ink
        overlapWord=self.overlap(mu=self.config)#compute overlap with congruent color of word
        overlapVec=np.zeros((len(self.congru),1))
        
        for jj in range(len(self.congru)):
            overlapVec[jj] = self.overlap(mu=jj) # with all the congruent colors
            
        if overlapInk >= overlapWord:# if the ink color is more retrieved than the word one
            if overlapInk==max(overlapVec):#we check if the ink color is actually the retrieved
                                #color                
                error=0
            
            else: #definitions of errordebile
                otherPattern=np.where(overlapVec==max(overlapVec))[0][0] #we identify which is the other color 
                print("Retrived different congruent pattern. Overlap with congruent pattern num {}" # that has been
                .format(otherPattern))  #retrieved
                print("Configuration of the wordcolor in the initial incongruent pattern is {}".format(self.config))
                
                error=-1
        else:
            error=1
            
        
        return [error, i_fin, overlapInk]