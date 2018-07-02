#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:53:53 2018

@author: Yohann
"""

import numpy as np
import scipy.stats as scs
import scipy.optimize as sco
import scipy.integrate as sci
import math

#%% Generate variables

def generate_variable(sample_size, sparsity=0, amplitude=0, sigma=1):
    N = 2*sample_size+1
    xi1 = scs.norm.rvs(0, 1, [N, ])
    xi2 = scs.norm.rvs(0, 1, [N, ])
    noise = sigma*(xi1+1j*xi2)
    res = generate_signal(sample_size, sparsity, amplitude)
    signal = res[0]
    y = signal + noise
    return [y, res[1], res[2], res[3]]

#%% Generate random signal

def generate_signal(sample_size, sparsity = 0, amplitude = 0):
    
    N   = 2*sample_size+1
    
    signal = np.zeros(N,)+1j*np.zeros(N,)
    support = 0
    phase = 0
    weight = 0
    
    if (sparsity>0) & (amplitude>0):
        support = np.random.uniform(0,2*np.pi,sparsity)
        phase   = np.random.uniform(0,2*np.pi,sparsity)
        weight   = amplitude*scs.norm.rvs(0,1,[sparsity,])*np.exp(-1j*phase)     
        for k in range(0,N):
            for t in range(0,sparsity):
                signal[k] += weight[t]*fourier(support[t], k)
                
    signal = signal/np.sqrt(N)   
    
    return [signal, support, phase, weight]

#%% Generate of sampled statistic

def generate_stat(sample_size, sparsity = 0, amplitude = 0, sigma = 1):
    
    """ 
    We generate variables and observation
    """
    var             = generate_variable(sample_size, sparsity, amplitude, sigma)
    y_obs           = var[0]
    
    """ 
    f is equal to -X(t,theta) and we will minimize f (max. X)
    """
    def f(x):
        """ 
        f(x)=-X(t,theta) where x[0]=t and x[1]=theta
        """
        res = np.real(np.exp(-1j*x[1])*\
                      sum(y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                          for k in range(-sample_size,sample_size+1)))       
        res = -res/np.sqrt(2*sample_size+1)    
        return res
    
    def grad_f(x):
        """ 
        gradient of f
        """
        res1 = np.real(np.exp(-1j*x[1])*\
                       sum(1j*k*y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                           for k in range(-sample_size,sample_size+1)))
        res1 = -res1/np.sqrt(2*sample_size+1)
        
        res2 = np.real(np.exp(-1j*x[1])*\
                       sum(-1j*y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                           for k in range(-sample_size,sample_size+1)))
        res2 = -res2/np.sqrt(2*sample_size+1)
        return np.array([res1, res2])
    
    #% Minimizing f
        
    """ 
    we minimize on [0, 2pi]^2
    """
    bnds    = ((0, 2*np.pi), (0, 2*np.pi))
    
    """ 
    We begin by a greedy search of the initialization point over a grid of size 126^2
    the initialization point is init
    """
    x  = y = np.arange(0, 2*np.pi, 0.05)
    steps   = 126
    X, Y    = np.meshgrid(x, y)
    val     = np.array([f([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    init    = np.argmin(val)
    x1      = init%steps
    x2      = (init-x1)/steps
    init    = [x1*0.05, x2*0.05]
    
    """ 
    we minimize f...
    """
    result  = sco.minimize(f, init, method="L-BFGS-B",\
                           jac=grad_f, bounds=bnds, tol=1e-15)
    
    """ 
    (t1,theta1) is the argmax of X(t, theta) and l1=$\lambda_1$
    """
    t1      = result.x[0]
    theta1  = result.x[1]
    l1      = -f([t1,theta1])
    
    
    """ 
    Function g(x) is equal to (X(t1,theta1)-X(x))/(1-rho((t1,theta1)-x))
    """
    def g(x):
        a0  = x[0]-t1
        a1  = x[1]-theta1
        N = 2*sample_size+1
        
        vec         = np.array([a0,a1])
        r           = np.linalg.norm(vec)
        """ 
        the value for r=0 is set to l1 (note that r=0 corresponds to x=(t1,theta1))
        """   
        res = l1 
    
        if (0<r) & (r<0.00001):
            """ 
            we look a values near (t1,theta1) for which an indetermination occurs
            """        
            alpha= np.arccos(np.clip(a0/np.sqrt(a0**2+a1**2), -1.0, 1.0))
            u0 = np.cos(alpha)
            u1 = np.sin(alpha)
            """ 
            u0,u1 defines the direction (unit vector)
            """
            denom  = sum((k*np.cos(alpha)-np.sin(alpha))**2*\
                   (np.sinc((r*(k*np.cos(alpha)-np.sin(alpha)))/(2*np.pi)))**2\
                   for k in range(-sample_size,sample_size+1))/N
            """ 
            denom computes the denominator
            """
            
#            """ 
#            We use simpson rule for the numerator
#            """
#            h       = np.linspace(0,1,500)
#            
#            b0      = t1 + h*a0
#            b1      = theta1 + h*a1
#            
#            value   = (1-h)*(u0**2*\
#                        np.real(np.exp(-1j*b1)*sum(-k**2*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))\
#                        +2*u0*u1*\
#                        np.real(np.exp(-1j*b1)*sum(k*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))\
#                        +u1**2*\
#                        np.real(np.exp(-1j*b1)*sum((-1)*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))) 
#            value   = value/np.sqrt(N)
#            
#            num = sci.simps(value, h)
    
            """ 
            we use a quadrature for the numerator
            """   
            fun_int = lambda w: (1-w)*(u0**2*\
                        np.real(np.exp(-1j*(theta1+w*a1))*\
                            sum(-k**2*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                for k in range(-sample_size,sample_size+1)))\
                        +2*u0*u1*\
                        np.real(np.exp(-1j*(theta1+w*a1))*\
                            sum(k*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                for k in range(-sample_size,sample_size+1)))\
                        +u1**2*\
                        np.real(np.exp(-1j*(theta1+w*a1))*\
                            sum((-1)*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                for k in range(-sample_size,sample_size+1)))) 
            
            num = np.mean(sci.quad(fun_int, 0, 1, epsabs=1e-15, epsrel=1e-15, limit=1000))
            
            res = -num/denom
    
        if (r>=0.00001):
            """ 
            we look a values far (t1,theta1) for which there is no indetermination
            """   
            res = (l1+f(x))/(1-(np.cos(a1)*dirichlet(a0,N)/N))
            
        return res    
    """ 
    we minimize g on [0, 2pi]^2 an dwe llok for the initialization point
    """
    val2    = np.array([g([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    init2   = np.argmin(val2)
    x1      = init2%steps
    x2      = (init2-x1)/steps
    init2   = [x1*0.05, x2*0.05]    
    result2 = sco.minimize(g, init2, method="L-BFGS-B", bounds=bnds, tol=1e-15)    
    """ 
    argmin of g
    """
    t2      = result2.x[0]
    theta2  = result2.x[1]    
    """ 
    value of lambda_2
    """
    l21     = l1-result2.fun    
    a0      = t2-t1
    a1      = theta2-theta1
    N       = 2*sample_size+1
    l22     = l1-(l1+f([t2,theta2]))/(1-(np.cos(a1)*dirichlet(a0,N)/N))
    l2      = max(l21,l22)
    """ 
    we compute the statistic
    """
    alpha1  = (1/3)*sample_size*(sample_size+1)
    alpha2  = (1/np.sqrt(N))*\
                sum((k**2-alpha1)*\
                    np.real(y_obs[k+sample_size]*np.exp(1j*(k*t1-theta1))) \
                    for k in range(-sample_size,sample_size+1))
    alpha3  = (1/np.sqrt(N))*sum(k*np.real(y_obs[k+sample_size]*\
                  np.exp(1j*(k*t1-theta1))) for k in range(-sample_size,sample_size+1))    
    stat    = (sigma*(alpha1*l1+alpha2)*scs.norm.pdf(l1/sigma)+\
               (alpha1*sigma**2-alpha3**2)*(1-scs.norm.cdf(l1/sigma)))/\
               (sigma*(alpha1*l2+alpha2)*scs.norm.pdf(l2/sigma)+\
                (alpha1*sigma**2-alpha3**2)*(1-scs.norm.cdf(l2/sigma))) 
               
    return stat
    
#%% Generate of sampled statistic when the level of noise is unknown
""" 
We need a noise level as input BUT we won't use it in the expression of the statistic
"""
def generate_tstat(sample_size, sparsity = 0, amplitude = 0, sigma = 1):
    
    m = 4*sample_size+2 # order of the kernel
    """ 
    We generate variables and observation
    """
    var             = generate_variable(sample_size, sparsity, amplitude, sigma)
    y_obs           = var[0]
    
    """ 
    f is equal to -X(t,theta) and we will minimize f (max. X)
    """
    def f(x):
        """ 
        f(x)=-X(t,theta) where x[0]=t and x[1]=theta
        """
        res = np.real(np.exp(-1j*x[1])*sum(y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                                                for k in range(-sample_size,sample_size+1)))       
        res = -res/np.sqrt(2*sample_size+1)    
        return res
    
    def grad_f(x):
        """ 
        gradient of f
        """
        res1 = np.real(np.exp(-1j*x[1])*sum(1j*k*y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                                                for k in range(-sample_size,sample_size+1)))
        res1 = -res1/np.sqrt(2*sample_size+1)
        
        res2 = np.real(np.exp(-1j*x[1])*sum(-1j*y_obs[k+sample_size]*np.exp(1j*k*x[0]) \
                                                for k in range(-sample_size,sample_size+1)))
        res2 = -res2/np.sqrt(2*sample_size+1)
        return np.array([res1, res2])
    
    #% Minimizing f
        
    """ 
    we minimize on [0, 2pi]^2
    """
    bnds    = ((0, 2*np.pi), (0, 2*np.pi))
    
    """ 
    We begin by a greedy search of the initialization point over a grid of size 126^2
    the initialization point is init
    """
    x  = y = np.arange(0, 2*np.pi, 0.05)
    steps   = 126
    X, Y    = np.meshgrid(x, y)
    val     = np.array([f([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    init    = np.argmin(val)
    x1      = init%steps
    x2      = (init-x1)/steps
    init    = [x1*0.05, x2*0.05]
    
    """ 
    we minimize f...
    """
    result  = sco.minimize(f, init, method="L-BFGS-B", jac=grad_f, bounds=bnds, tol=1e-15)
    
    """ 
    (t1,theta1) is the argmax of X(t, theta) and l1=$\lambda_1$
    """
    t1      = result.x[0]
    theta1  = result.x[1]
    l1      = -f([t1,theta1])
    
    
    """ 
    Function g(x) is equal to (X(t1,theta1)-X(x))/(1-rho((t1,theta1)-x))
    """
    def g(x):
        a0  = x[0]-t1
        a1  = x[1]-theta1
        N = 2*sample_size+1
        
        vec         = np.array([a0,a1])
        r           = np.linalg.norm(vec)
        """ 
        the value for r=0 is set to l1 (note that r=0 corresponds to x=(t1,theta1))
        """   
        res = l1 
    
        if (0<r) & (r<0.00001):
            """ 
            we look a values near (t1,theta1) for which an indetermination occurs
            """        
            alpha= np.arccos(np.clip(a0/np.sqrt(a0**2+a1**2), -1.0, 1.0))
            u0 = np.cos(alpha)
            u1 = np.sin(alpha)
            """ 
            u0,u1 defines the direction (unit vector)
            """
            denom       = sum((k*np.cos(alpha)-np.sin(alpha))**2*\
                               (np.sinc((r*(k*np.cos(alpha)-np.sin(alpha)))/(2*np.pi)))**2\
                               for k in range(-sample_size,sample_size+1))/N
            """ 
            denom computes the denominator
            """
            
#            """ 
#            We use simpson rule for the numerator
#            """
#            h       = np.linspace(0,1,500)
#            
#            b0      = t1 + h*a0
#            b1      = theta1 + h*a1
#            
#            value   = (1-h)*(u0**2*\
#                        np.real(np.exp(-1j*b1)*sum(-k**2*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))\
#                        +2*u0*u1*\
#                        np.real(np.exp(-1j*b1)*sum(k*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))\
#                        +u1**2*\
#                        np.real(np.exp(-1j*b1)*sum((-1)*y_obs[k+sample_size]*np.exp(1j*k*b0) \
#                                                    for k in range(-sample_size,sample_size+1)))) 
#            value   = value/np.sqrt(N)
#            
#            num = sci.simps(value, h)
    
            """ 
            we use a quadrature for the numerator
            """   
            fun_int = lambda w: (1-w)*(u0**2*\
                        np.real(np.exp(-1j*(theta1+w*a1))*sum(-k**2*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                    for k in range(-sample_size,sample_size+1)))\
                        +2*u0*u1*\
                        np.real(np.exp(-1j*(theta1+w*a1))*sum(k*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                    for k in range(-sample_size,sample_size+1)))\
                        +u1**2*\
                        np.real(np.exp(-1j*(theta1+w*a1))*sum((-1)*y_obs[k+sample_size]*np.exp(1j*k*(t1+w*a0)) \
                                                    for k in range(-sample_size,sample_size+1)))) 
            
            num = np.mean(sci.quad(fun_int, 0, 1, epsabs=1e-15, epsrel=1e-15, limit=1000))
            
            res = -num/denom
    
        if (r>=0.00001):
            """ 
            we look a values far (t1,theta1) for which there is no indetermination
            """   
            res = (l1+f(x))/(1-(np.cos(a1)*dirichlet(a0,N)/N))
            
        return res    
    """ 
    we minimize g on [0, 2pi]^2 an dwe llok for the initialization point
    """
    x  = y = np.arange(0, 2*np.pi, 0.05)
    steps   = 126
    X, Y    = np.meshgrid(x, y)
    val2    = np.array([g([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    init2   = np.argmin(val2)
    x1      = init2%steps
    x2      = (init2-x1)/steps
    init2   = [x1*0.05, x2*0.05]    
    result2 = sco.minimize(g, init2, method="L-BFGS-B", bounds=bnds, tol=1e-15)    
    """ 
    argmin of g
    """
    t2      = result2.x[0]
    theta2  = result2.x[1]    
    """ 
    value of lambda_2
    """
    l21     = l1-result2.fun    
    a0      = t2-t1
    a1      = theta2-theta1
    N       = 2*sample_size+1
    l22     = l1-(l1+f([t2,theta2]))/(1-(np.cos(a1)*dirichlet(a0,N)/N))
    l2      = max(l21,l22)
    
    """ 
    we compute the noise level estimation
    """    
    
    """ 
    we start by defining the normalized regressed process $X^{|}_{norm}$
    referred to as h(x) in this code
    """
    alpha1  = (1/3)*sample_size*(sample_size+1)
    alpha2  = (1/np.sqrt(N))*\
                sum((k**2-alpha1)*\
                    np.real(y_obs[k+sample_size]*np.exp(1j*(k*t1-theta1))) \
                    for k in range(-sample_size,sample_size+1))
    alpha3  = (1/np.sqrt(N))*sum(k*np.real(y_obs[k+sample_size]*\
                  np.exp(1j*(k*t1-theta1))) for k in range(-sample_size,sample_size+1)) 
    def rho(x):
        return np.cos(x[1])*dirichlet(x[0],N)/N
    
    def rho_grad(x):
        res1 = -np.cos(x[1])*(1/N)*(2*\
                      sum(k*np.sin(k*x[0]) for k in range(1,sample_size+1)))
        res2 = -np.sin(x[1])*(1/N)*dirichlet(x[0],N)
        return np.array([res1,res2])
    
    def h(x):
        z = [t1,theta1] # argmax of X
        r = rho_grad(z-x)
        num = -f(x)-rho(z-x)*l1
        denom = np.sqrt(\
                        1-rho(z-x)**2\
                        +((1/alpha1)*r[0]**2+r[1]**2)
                        )
        return num/denom
    
    """ 
    we estimate sigma on npts random points 
    (m-1 suffices with large probability) 
    """ 
    npts    = 4*m
    pts1    = np.random.uniform(0,2*np.pi,npts)
    pts2    = np.random.uniform(0,2*np.pi,npts)
    values  = np.zeros(npts)
    cov     = np.zeros((npts,npts))
    z       = np.array([t1,theta1])
    
    for k in range(0,npts):
        for l in range(0,npts):
            y1 = np.array([pts1[k],pts2[k]])
            y2 = np.array([pts1[l],pts2[l]])
            r1 = rho(y1-y2)
            r11= rho(z-y1)
            r12= rho(z-y2)
            g11= rho_grad(z-y1)
            g12= rho_grad(z-y2)
            num = r1-r11*r12+((1/alpha1)*g11[0]*g12[0]+g11[1]*g12[1])
            denom1 = np.sqrt(1-rho(z-y1)**2+((1/alpha1)*g11[0]**2+g11[1]**2))
            denom2 = np.sqrt(1-rho(z-y2)**2+((1/alpha1)*g12[0]**2+g12[1]**2))  
            cov[k,l] = num/(denom1*denom2)
            
    for k in range(0,npts):
        values[k] = h(np.array([pts1[k],pts2[k]]))
        
    w, v = np.linalg.eigh(cov)
    
    w = np.abs(w)*(w>1e-10)
    
    inv_sqrt_w = np.zeros(npts)
    
    for k in range(0,npts):
        if w[k]!=0:
            inv_sqrt_w[k]=1/np.sqrt(w[k])
            
    
    white = np.dot(np.transpose(values),np.dot(v,np.diag(inv_sqrt_w )))
    
    k2 = np.linalg.norm(white)**2
    
    var_estimated = k2/(m-3)
    
    std_estimated = np.sqrt(var_estimated)
    
    T1 = l1/std_estimated
    T2 = l2/std_estimated
    g0 = ((m-3)/(m-2))*(math.gamma(m/2)*math.gamma((m-3)/2))/(math.gamma((m-1)/2)*math.gamma((m-2)/2))
    
    df1=m-3
    df2=m-1
    
    rv1 = scs.t(df1)
    rv2 = scs.t(df2)
    
    """ 
    we compute the statistic
    """
   
    stat    = (alpha1*(1-rv1.cdf(T1))+(alpha1*T1+alpha2)*rv1.pdf(T1)-g0**(-1)*alpha3**2*(1-rv2.cdf(T1)))/\
            (alpha1*(1-rv1.cdf(T2))+(alpha1*T2+alpha2)*rv1.pdf(T2)-g0**(-1)*alpha3**2*(1-rv2.cdf(T2))) 
               
    return [stat, var_estimated]

#%% Dirichlet kernel

def dirichlet(x,N):
    if x==0:
        t=N
    else:
        t = np.sin(N*x/2)/np.sin(x/2)
    return t

#%% Fourier curve
    
def fourier(x,k):
    return np.exp(-1j*k*x)
