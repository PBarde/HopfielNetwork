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


class HopfieldNetwork:
    """Implements a Hopfield network of size N.

    Attributes:
        N (int): Number of neurons
        patterns (numpy.ndarray): Array of stored patterns
        weight (numpy.ndarray): Array of stored weights
        x (numpy.ndarray): Network state (of size N**2)
    """

    def __init__(self, N):
        self.N = N

    def make_pattern(self, P=1, ratio=0.5):
        """Creates and stores additional patterns to the
        network.

        Args:
            P (int, optional): number of patterns
                (used only for random patterns)
            ratio (float, optional): percentage of 'on' pixels
                for random patterns, so white pixels

        Raises:
            ValueError: Raised if N!=10 and letters!=None. For now
                letters are hardcoded for N=10.
        """
        
        self.patterns = -np.ones((P, self.N), int)
        idx = int(ratio*self.N) #number of 'on' pixels 
        for i in range(P):
            self.patterns[i, :idx] = 1 #we turn 'on' the idx first pixels
                #of each pattern
            self.patterns[i] = np.random.permutation(self.patterns[i])
                #randomly permutes the 'on' and 'off' pixels in each pattern
        self.weight = np.zeros((self.N, self.N))
        for i in range(self.N): #for each pixel
            self.weight[i] = 1./self.N * (
                np.sum(
                    self.patterns[k, i] * self.patterns[k]
                    for k in range(self.patterns.shape[0]) #k runs on the number
                    #of patterns
                )
            )
            self.weight[i,i]=0 #we impose no autaptic connections


    def dynamic(self):
        """Executes one timestep of the dynamics"""
        idx=range(self.N)
        idx=np.random.permutation(idx)#asychronous update with random order
        h=np.ones(self.N)        
        for ii in idx:
            h[ii]=np.sum(self.weight*self.x,axis=1)[ii]
            
            if h[ii]==0:
                self.x[ii]=1;
            else:
                self.x[ii]=np.sign(h[ii])
                
                
    def overlap(self, mu):
        """Computes the overlap of the current state with
        pattern number mu.

        Args:
            mu (int): The index of the pattern to
                compare with. mu<P
        """

        return 1./self.N*np.sum(self.patterns[mu]*self.x)
        
        
        
    def retrival_error(self,mu):
        """Computes the retrival error of the current state with the pattern
        number mu
        
        Args: 
            mu (int): The index of the pattern to
                compare with. mu<P
        """
        
        return 1./self.N*np.sum((1-self.patterns[mu]*self.x)/2)
        

    def run(self, t_max=20, mu=0, flip_ratio=0):
        """Runs the dynamics and optionally plots it.

        Args:
            t_max (float, optional): Timesteps to simulate
            mu (int, optional): Pattern number to use
                as initial pattern for the network state (< P)
            flip_ratio (int, optional): ratio of randomized pixels.
                For example, to run pattern #5 with 5% flipped pixels use
                ``run(mu=5,flip_ratio=0.05)``
        Raises:
            IndexError: Raised if given pattern index is too high.
            RuntimeError: Raised if no patterns have been created.
        """
        try:
            self.patterns
        except AttributeError:
            raise RuntimeError(
                'No patterns created: please ' +
                'use make_pattern to create at least one pattern.'
            )

        try:
            self.patterns[mu]
        except:
            raise IndexError('Pattern index too high (has to be < P)')

        # set the initial state of the net
        self.x = copy(self.patterns[mu])  #we start from the chosen pattern
        flip = np.random.permutation(np.arange(self.N)) #we randomly select 
        #the order of the pixels to be flipped
        idx = int(self.N * flip_ratio) #total number of pixels to be flipped
        self.x[flip[0:idx]] *= -1 #we multiply by -1 (we flipped) the pixels
        
        t = [0] #vector of times
        overlap = [self.overlap(mu)]#vector of overlaps, different from the 
        #function
        
        x_old = copy(self.x)

        for i in range(t_max):

            # run a step
            self.dynamic()
            t.append(i+1)
            overlap.append(self.overlap(mu))

            # check the exit condition
            # if there is no change in the network we stop the simulation 
            i_fin = i
            if np.sum(np.abs(x_old-self.x)) == 0:
                break
            x_old = copy(self.x)

        print("Pattern recovered in %i time steps." % i_fin +
              " Final overlap %.3f" % overlap[-1])
        return self.retrival_error(mu=mu)

