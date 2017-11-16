# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:41:44 2016

@author: paul
"""

#%%
import matplotlib
#The following lines define the plot aspects
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('axes', labelsize=22, titlesize=26) 
matplotlib.rc('lines', lw=2) 

from hopfield_ex2 import HopfieldNetwork
import matplotlib.pyplot as plt
import numpy as np
from copy import copy #creates a copy of the entity at a new adress
import random as rand

#%% Question 2.1
error=-10
i_fin=-11
n=HopfieldNetwork(100)
P=5
n.make_pattern(P=P, k=0.5)

mu=rand.randint(0,P-1)

[error, i_fin, overlapInk] = n.run(mu=mu,typepattern='incongru')

if error==0:
    print('Inkcolor of initialized pattern {} found in {} iterations'.format(mu,i_fin))
    
elif error==1:
    print('Did not found Inkcolor of initialized pattern {} but converged in {} iterations'.format(mu,i_fin))
    
else:
    print('Retrived a different color starting from incongruent with ink color {} and word color {} in {} iterations'.format(mu,n.config,i_fin))


#%% Question 2.2
N=100
P=5
Nrealization=40
Ntrials=60

kvec=np.arange(0.3,0.8,0.05)

meanOverlapVec=np.zeros(len(kvec))
meanReactionTime=np.zeros(len(kvec))
meanError=np.zeros(len(kvec))
meanErrorDebile=np.zeros(len(kvec))

confOverlapVec=np.zeros(len(kvec))
confReactionTime=np.zeros(len(kvec))
confError=np.zeros(len(kvec))
confErrorDebile=np.zeros(len(kvec))

for kk in range(len(kvec)): #we do a loop on the percentage bits 
    
    k=kvec[kk]
    overlapVec=[]
    reactionTimeVec=[]
    errorVec=[]
    errorDebileVec=[]

    for nn in range(Nrealization): #for each of this percentage bits we take Nrealization
        n=HopfieldNetwork(N=N)     # or different subjects
        n.make_pattern(P=P,k=k)    # each of these subjects learns a set of patterns
        errorCount=0
        errorDebileCount=0
        
        for ii in range(Ntrials): # to each of these subjects we do Ntrials where 
                                  # we show them a different incongruent word
            muRand=rand.randint(0,P-1)# we start with a randomly chosen ink color
            [error, i_fin, overlapInk]=n.run(mu=muRand,typepattern='incongru')
            
            overlapVec.append(overlapInk)
            reactionTimeVec.append(i_fin)
            
            if error!=-1:
                errorCount=errorCount+error
            
            else:
                errorCount=errorCount+1
                errorDebileCount=errorDebileCount+1
        
        errorVec.append(errorCount)
        errorDebileVec.append(errorDebileCount)
    
    meanOverlapVec[kk]=np.mean(overlapVec)
    meanReactionTime[kk]=np.mean(reactionTimeVec)
    meanError[kk]=np.mean(errorVec)
    meanErrorDebile[kk]=np.mean(errorDebileVec)
    
    confOverlapVec[kk]=np.std(overlapVec)/np.sqrt(Ntrials*Nrealization)*1.96
    confReactionTime[kk]=np.std(reactionTimeVec)/np.sqrt(Ntrials*Nrealization)*1.96
    confError[kk]=np.std(errorVec)/np.sqrt(Nrealization)*1.96
    confErrorDebile[kk]=np.std(errorDebileVec)/np.sqrt(Nrealization)*1.96    
#%%
fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(kvec,meanOverlapVec,confOverlapVec)
ax.grid(True)
plt.xlabel('Percentage of color bits')
plt.ylabel('Mean overlap with Inkcolor congruent pattern')
plt.title('Error bar represent 95% confidence intervals')

fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(kvec,meanReactionTime,confReactionTime)
ax.grid(True)
plt.xlabel('Percentage of color bits')
plt.ylabel('Mean reaction time')
plt.title('Error bar represent 95% confidence intervals')


fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(kvec,meanError,confError)
ax.grid(True)
plt.xlabel('Percentage of color bits')
plt.ylabel('Mean number of errors for each realization')
plt.title('Error bar represent 95% confidence intervals')

fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(kvec,meanErrorDebile,confErrorDebile)
ax.grid(True)
plt.xlabel('Percentage of color bits')
plt.ylabel('Mean number of UNEXPECTED errors for each realization')
plt.title('Error bar represent 95% confidence intervals')
#%% Question 2.3
N=100
Nrealization=40
Ntrials=60
k=0.6

Pvec=np.arange(2,20,2)

meanOverlapVec=np.zeros(len(Pvec))
meanReactionTime=np.zeros(len(Pvec))
meanError=np.zeros(len(Pvec))
meanErrorDebile=np.zeros(len(Pvec))

confOverlapVec=np.zeros(len(Pvec))
confReactionTime=np.zeros(len(Pvec))
confError=np.zeros(len(Pvec))
confErrorDebile=np.zeros(len(Pvec))

for kk in range(len(Pvec)):#Same as previous question but here be loop on the
                            #number of colors for a fixed inkcolor bits percentage
    P=Pvec[kk]
    overlapVec=[]
    reactionTimeVec=[]
    errorVec=[]
    errorDebileVec=[]

    for nn in range(Nrealization):
        n=HopfieldNetwork(N=N)
        n.make_pattern(P=P,k=k)
        errorCount=0
        errorDebileCount=0
        
        for ii in range(Ntrials):
            
            muRand=rand.randint(0,P-1)
            [error, i_fin, overlapInk]=n.run(mu=muRand,typepattern='incongru')
            
            overlapVec.append(overlapInk)
            reactionTimeVec.append(i_fin)
            
            if error!=-1:
                errorCount=errorCount+error
            
            else:
                errorCount=errorCount+1
                errorDebileCount=errorDebileCount+1
        
        errorVec.append(errorCount)
        errorDebileVec.append(errorDebileCount)
    
    
    meanOverlapVec[kk]=np.mean(overlapVec)
    meanReactionTime[kk]=np.mean(reactionTimeVec)
    meanError[kk]=np.mean(errorVec)
    meanErrorDebile[kk]=np.mean(errorDebileVec)
    
    confOverlapVec[kk]=np.std(overlapVec)/np.sqrt(Ntrials*Nrealization)*1.96
    confReactionTime[kk]=np.std(reactionTimeVec)/np.sqrt(Ntrials*Nrealization)*1.96
    confError[kk]=np.std(errorVec)/np.sqrt(Nrealization)*1.96
    confErrorDebile[kk]=np.std(errorDebileVec)/np.sqrt(Nrealization)*1.96 
    
#%%
fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(Pvec,meanOverlapVec,confOverlapVec)
ax.grid(True)
plt.xlabel('Number of stored colors')
plt.ylabel('Mean overlap with Inkcolor congruent pattern')
plt.title('Error bar represent 95% confidence intervals')

fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(Pvec,meanReactionTime,confReactionTime)
ax.grid(True)
plt.xlabel('Number of stored colors')
plt.ylabel('Mean reaction time')
plt.title('Error bar represent 95% confidence intervals')

fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(Pvec,meanError,confError)
ax.grid(True)
plt.xlabel('Number of stored colors')
plt.ylabel('Mean number of errors for each realization')
plt.title('Error bar represent 95% confidence intervals')   

fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(Pvec,meanErrorDebile,confErrorDebile)
ax.grid(True)
plt.xlabel('Percentage of color bits')
plt.ylabel('Mean number of UNEXPECTED errors for each realization')
plt.title('Error bar represent 95% confidence intervals')
#%% Question 2.4

N=100
P=5
k=0.5

n=HopfieldNetwork(N=N)
n.make_pattern(P=P,k=k)

mu=rand.randint(0,P-1)
#%% we ran this section independently to initialize with the same ink color
# to have always the same initial incongruent pattern we did this with P=2
[error, i_fin, overlapInk] = n.run(mu=mu,typepattern='incongru', update='synch')

if error==0:
    print('Inkcolor of initialized pattern {} found in {} iterations'.format(mu,i_fin))
    
elif error==1:
    print('Did not found Inkcolor of initialized pattern {} but converged in {} iterations'.format(mu,i_fin))
    
else:
    print('Retrived a different color starting from Inkcolor {} and word color {} in {} iterations'.format(mu,n.config,i_fin))


