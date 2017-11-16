import matplotlib
matplotlib.use('TkAgg')
from hopfield_ex1 import HopfieldNetwork
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import pickle
import gzip
import random as rand




##%%
#n=HopfieldNetwork(200)
#
#n.make_pattern(P=5)
#n.run(flip_ratio=0.1)
##%%
#n.run(flip_ratio=0.3)
#%% Question 1.3.a
P=5
n=HopfieldNetwork(200)

# First loop on flip frequency
flip_ratioVector=np.arange(0.01,0.51,0.01)
Nrealization=40
meanRetrivalError=[]
VariabilityRetrivalError=[]
errorbar=0
deviation=0
mean=0

for flip_ratio in flip_ratioVector:
    RetrivalError=np.zeros(Nrealization)
    
    for ii in range(Nrealization):
        n.make_pattern(P=P)
        mu=rand.randint(0,P-1)
        RetrivalError[ii]=n.run(mu=mu, flip_ratio=flip_ratio)
        
    mean=np.mean(RetrivalError)
    #deviation=np.sum(abs((RetrivalError-mean)/mean))
    deviation=np.std(RetrivalError)
    errorbar=deviation/np.sqrt(Nrealization)*1.96#to have 95% confidence interval
    # we could also define the error as (mean-RetrivalError)/mean and sum it
    meanRetrivalError.append(mean)
    VariabilityRetrivalError.append(errorbar)
    

#%%
fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(flip_ratioVector,meanRetrivalError,VariabilityRetrivalError,lw=1.5)
#ax.xaxis.grid(True)
#ax.yaxis.grid(True)
ax.grid(True)
plt.xlabel('Fraction of fliped bits',fontsize=16)
plt.ylabel('Mean retrieval error',fontsize=16)
plt.title('Error bar represent 95% confidence intervals',fontsize=20)
##Put figure window on top of all other windows
#fig.canvas.manager.window.attributes('-topmost', 1)
##After placing figure window on top, allow other windows to be on top of it later
#fig.canvas.manager.window.attributes('-topmost', 0)

#%% Question 1.3.b

n=HopfieldNetwork(200)

flip_ratio=0.1
P_vec=np.arange(1,30)
Nrealization=100
meanRetrivalError=[]
VariabilityRetrivalError=[]
errorbar=0
deviation=0
mean=0

for P in P_vec:
    RetrivalError=np.zeros(Nrealization)
    
    for ii in range(Nrealization):
        n.make_pattern(P=P)
        mu=rand.randint(0,P-1)
        RetrivalError[ii]=n.run(mu=mu, flip_ratio=flip_ratio)
        
    mean=np.mean(RetrivalError)
    #deviation=np.sum(abs((RetrivalError-mean)/mean))
    deviation=np.std(RetrivalError)
    errorbar=deviation/np.sqrt(Nrealization)*1.96#to have 95% confidence interval
    # we could also define the error as (mean-RetrivalError)/mean and sum it
    meanRetrivalError.append(mean)
    VariabilityRetrivalError.append(errorbar)
    
#%%
fig = plt.figure()
ax=fig.gca()
g1=plt.errorbar(P_vec,meanRetrivalError,VariabilityRetrivalError,lw=1.5)
#ax.xaxis.grid(True)
#ax.yaxis.grid(True)
ax.grid(True)
plt.xlabel('Number of stored patterns',fontsize=16)
plt.ylabel('Mean retrieval error',fontsize=16)
plt.title('Error bar represent 95% confidence intervals',fontsize=20)
##Put figure window on top of all other windows
#fig.canvas.manager.window.attributes('-topmost', 1)
##After placing figure window on top, allow other windows to be on top of it later
#fig.canvas.manager.window.attributes('-topmost', 0)    
#%% Question 1.4
n=HopfieldNetwork(500)

Nrealization=10
flip_ratio=0.1
VariabilityRetrivalError=[]
confidence=0
deviation=0
maxLoadVector=[]

for ii in range(Nrealization): #repeat the retrieval of paterns Nrealization times
    mean=0
    P=55
    condition=0
    print('ii=%i' %ii)
    while condition==0: #if the mean retrieval error averaged over all paterns 
                        #is below 2%
        P+=1            #each time we try with one more pattern
        print('P=%i' %P)
        muVector=range(P)
        n.make_pattern(P=P)
        RetrivalError=np.zeros(P)
    
        for mu in muVector: #we try to retrieve each patern one after the other
            print('mu=%i' %mu)
            RetrivalError[mu]=n.run(mu=mu, flip_ratio=flip_ratio,t_max=40)
        
        mean=np.mean(RetrivalError) 
        
        if mean > 0.02:
            condition=1
            
    neurons=float(n.N) #need to have a float for the following division 
                       #otherwise we end up with zero
    maxLoadVector.append(P/neurons)
    
meanLoad=np.mean(maxLoadVector)
deviation=np.std(maxLoadVector)
confidence=deviation/np.sqrt(Nrealization)*1.96

print('The maximal load for a network of {} neurons and a flip ratio of {} is {}. The 95% \
confidence interval is {}'\
    .format(n.N , flip_ratio , meanLoad, confidence))
#meanLoad=  0.12663157894736843
#confidence=0.001707566994747571
