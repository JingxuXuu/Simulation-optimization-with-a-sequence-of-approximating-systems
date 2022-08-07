#This file provides the experiment for simulation-optimization algorithm based on FD estimator
import numpy as np
obj=2
d=5


S0=0.5
r=0.1

def generateS(n,M,B,mu):
    S=S0*np.ones(d)
    delta=1/(M**n)
    wt=np.random.normal(0, delta**(0.5), (M**n,d))
    
    for i in range(M**n):
        for j in range(d):
            S[j]=S[j]+ mu[j]*S[j]*delta + S[j]*np.dot(B[j],wt[i])
        
    return S,(M**n)*d


def approxsgd(M, gamma_0,N_0,m_0, x_0, beta, rho, r, t, B,mu):
    
    
    x=x_0
    complexity=0
    complexity_list=np.zeros(t)
    loss=np.zeros(t)
    S1= S0*np.exp(r)
    for i in range(1,t+1):          # Algorithm iteration
        approxcumgrad=np.zeros(d)
        h= (d**(-3/2))*(i**(-rho))
        for j in range(int(np.ceil(N_0*(d**5)*(i**(r+rho*2))))):
            Zorg=np.random.normal(0,1,d)
                           
                     
            dev= np.sqrt(np.sum(Zorg**2, axis=0))   
            Z = np.sqrt(d)*Zorg/dev
            
            S,runcomplex=generateS(int(np.ceil(4*np.log(m_0*d*(i**rho))/np.log(M))),M,B,mu)
            Sreg=S-S1*np.ones(d)
            currentfunctionvalue= (np.dot(Sreg,x)+S1-obj)**2
            onestepfunctionvalue= (np.dot(Sreg,x+h*Z)+S1-obj)**2
            currentapproxgrad= Z*(onestepfunctionvalue-currentfunctionvalue)/h
                  
            approxcumgrad=approxcumgrad + currentapproxgrad
            complexity=complexity + runcomplex
        approxgrad= (1/int(np.ceil(N_0*(i**(r)))))*approxcumgrad
        complexity_list[i-1]=complexity
        x = x - gamma_0* approxgrad / (i**beta)
        for j in range(d):
            if x[j] < 0:
                x[j]=0
        if np.sum(x)>1:
            x = x/np.sum(x)
        
        optimal=np.array([0.1676141,  0.12145978, 0.15933705, 0.18361757, 0.14116673]) #the optimum point for the problem
        loss[i-1]=np.dot(x-optimal,x-optimal)
        
    return x, complexity_list,loss


complexity=np.zeros(200)
complexity=np.zeros(200)

#Set parameters for the experiment in Section 5
B=np.array([[1.       ,  0.08556837, 0.09872055, 0.16524606, 0.06954566],
 [0.14352845, 1.      ,   0.16425313, 0.08897677, 0.11689557],
 [0.15313661, 0.15265617, 1.      ,   0.08923094, 0.09646848],
 [0.13546676, 0.06201125, 0.11419366, 1.   ,      0.13244801],
 [0.13219382, 0.15286227, 0.05724685, 0.08891189, 1.        ]])
mu=np.array([0.74398551, 0.57917634, 0.68815336, 0.79120564, 0.56790853])

loss=0
for k in range(200):
    xfinal, complexity_final,add_loss=approxsgd(2, 10,1,5, np.zeros(d), 1, 1/2, 0, 200, B,mu)
    loss=loss+ add_loss
