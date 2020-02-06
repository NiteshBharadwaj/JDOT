# -*- coding: utf-8 -*-
"""
Regression example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import pylab as pl

import jdot

#from sklearn import datasets
import sklearn
from scipy.spatial.distance import cdist 
import ot


#%% data generation

seed=1985
np.random.seed(seed)

n = 200
ntest=200


def get_data(n,ntest):

    n2=int(n/2)
    sigma=0.05
    
    xs=np.random.randn(n,1)+2
    xs[:n2,:]-=4
    ys=sigma*np.random.randn(n,1)+np.sin(xs/2)
    
    
    
    xt=np.random.randn(n,1)+2
    xt[:n2,:]/=2 
    xt[:n2,:]-=3
      
    yt=sigma*np.random.randn(n,1)+np.sin(xt/2)
    yt = -yt
    xt-=8
    
    return xs,ys,xt,yt

xs,ys,xt,yt=get_data(n,ntest)

fs_s = lambda x: np.sin(x/2)
fs_t = lambda x: -np.sin((x+8)/2)

                       
xvisu=np.linspace(-10,10,100)

#pl.savefig('imgs/visu_data_reg.eps')

#%% TLOT
lambd0=1e1
itermax=15
gamma=1e-1
alpha=1e0/4
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,yt,metric='sqeuclidean')
C=alpha*C0+fcost
G0=ot.emd(ot.unif(n),ot.unif(n),C)

model,loss = jdot.jdot_krr(xs,ys,xt,gamma_g=gamma,numIterBCD = 10, alpha=alpha, lambd=lambd0,ktype='rbf')
model_ot,loss_ot = jdot.ot_krr(xs,ys,xt,G0,gamma_g=gamma,numIterBCD = 10, alpha=alpha, lambd=lambd0,ktype='rbf')

K=sklearn.metrics.pairwise.rbf_kernel(xt,xt,gamma=gamma)
Kvisu=sklearn.metrics.pairwise.rbf_kernel(xvisu.reshape((-1,1)),xt,gamma=gamma)
ypred=model.predict(Kvisu)
ypred0=model.predict(K)
ypred_ot = model_ot.predict(Kvisu)


# compute true OT
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,ypred0,metric='sqeuclidean')
C=alpha*C0+fcost
G=ot.emd(ot.unif(n),ot.unif(n),C)

pl.figure(2)
pl.scatter(xs,ys,edgecolors='k')
pl.scatter(xt,yt,edgecolors='k')
#pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
#pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
nb=15
fs=17
idv=np.random.permutation(n)
for i in range(nb):
    idt=G[idv[i],:].argmax()
    if not i:
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],color='C2',label='JDOT matrix link')
    else:    
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],color='C2')
for i in range(nb):
    idt=G0[idv[i],:].argmax()
    if not i:
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k--',label='OT matrix link')
    else:
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k--')


pl.xlabel('x')

#pl.ylabel('y')
pl.legend(loc=4)
pl.title('Joint OT matrices',fontsize=fs)
pl.show()

pl.figure(2)
pl.scatter(xs,ys,edgecolors='k')
pl.scatter(xt,yt,edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),label='Source model')
pl.plot(xvisu,fs_t(xvisu),label='Target model')
pl.plot(xvisu,ypred,'g',label='JDOT model')
pl.plot(xvisu,ypred_ot,'r',label='OT model')
pl.xlabel('x')
fs=17

#pl.ylabel('y')
pl.legend(loc=4,fontsize=.7*fs)
pl.title('Model estimated with JDOT',fontsize=fs)
pl.show()
pl.tight_layout(pad=00,h_pad=0)
#pl.savefig('imgs/visu_reg2.eps')
pl.savefig('imgs/visu_reg2.pdf')
pl.savefig('imgs/visu_reg2.png')