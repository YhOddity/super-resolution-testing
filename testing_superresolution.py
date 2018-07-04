#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:07:51 2018

@author: Yohann DE CASTRO (yohann.decastro "at" gmail)

This code reproduces some numerical experiments of the paper entitled

"Testing Gaussian Process with Applications to Super-Resolution"
by J.-M. Azaïs and Y. De Castro

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aux import generate_stat, generate_tstat

#%%

"""
The known variance case
"""

"""
We begin with the distribution of S under the null
"""

sample_size     = 5
sparsity        = 0 
amplitude       = 0
sigma           = 1
iterations      = 500

emp = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_stat(sample_size, sparsity, amplitude, sigma)     
    emp[k] = temp

np.save('fig1_1', [emp, sample_size, sparsity, amplitude, sigma, iterations])

"""
We continue with the distribution of S under an alternative 
"""

sparsity        = 1 
amplitude       = np.log(2*sample_size+1)

emp2 = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_stat(sample_size, sparsity, amplitude, sigma)     
    emp2[k] = temp

np.save('fig1_2', [emp2, sample_size, sparsity, amplitude, sigma, iterations])

"""
We continue with the distribution of S under an other alternative 
"""

sparsity        = 1 
amplitude       = np.sqrt(2*sample_size+1)

emp3 = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_stat(sample_size, sparsity, amplitude, sigma)     
    emp3[k] = temp

np.save('fig1_3', [emp3, sample_size, sparsity, amplitude, sigma, iterations])

"""
We plot the results 
"""

#%
sns.set_style("dark")
a =emp
b =emp2
c =emp3
plt.figure(figsize=(6,6))
x = np.linspace(0,1,2000)
plt.plot(x, x, color="dimgray", linestyle="--", lw=2)
plt.plot(np.sort(a), np.linspace(0, 1, len(a)), color="royalblue", lw=3)
plt.plot(np.sort(b), np.linspace(0, 1, len(b)), color="darkseagreen", lw=3)
plt.plot(np.sort(c), np.linspace(0, 1, len(c)), color="indianred", lw=3)
plt.grid()
plt.xlim(-0.01,1.01)
plt.ylim(-0.01,1.01)
#plt.xlabel("x")
#plt.ylabel("F(x)")
plt.legend(["CDF of Uniform Law", "CDF of $S$ under the null",\
            "CDF of $S$ under $\log(N)$ alt.", "CDF of $S$ under $\sqrt{N}$ alt."])
plt.title ("Emp. CDFs with $f_c=$%s over %s iterations" %(sample_size, iterations))
plt.tight_layout()
plt.savefig('fig1.png', format='png', dpi=300)
plt.show()

#%%

"""
The unknown variance case
"""

"""
We begin with the distribution of T under the null
"""

sample_size     = 5
sparsity        = 0 
amplitude       = 0
sigma           = 1
iterations      = 5000

emp = np.zeros(iterations)
var_emp = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_tstat(sample_size, sparsity, amplitude, sigma)     
    emp[k] = temp[0]
    var_emp[k] = temp[1]

np.save('fig2_1', [emp, var_emp, sample_size, sparsity, amplitude, sigma, iterations])

"""
We continue with the distribution of S under an alternative 
"""

sparsity        = 1 
amplitude       = np.log(2*sample_size+1)

emp2 = np.zeros(iterations)
var_emp2 = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_tstat(sample_size, sparsity, amplitude, sigma)     
    emp2[k] = temp[0]
    var_emp2[k] = temp[1]

np.save('fig2_2', [emp2, var_emp2, sample_size, sparsity, amplitude, sigma, iterations])

"""
We continue with the distribution of S under an other alternative 
"""

sparsity        = 1 
amplitude       = np.sqrt(2*sample_size+1)

emp3 = np.zeros(iterations)
var_emp3 = np.zeros(iterations)

for k in range(0,iterations):    
    temp   = generate_tstat(sample_size, sparsity, amplitude, sigma)     
    emp3[k] = temp[0]
    var_emp3[k] = temp[1]

np.save('fig2_3', [emp3, var_emp3, sample_size, sparsity, amplitude, sigma, iterations])

"""
We plot the results 
"""

#%
sns.set_style("dark")
a =emp
b =emp2
c =emp3
plt.figure(figsize=(6,6))
x = np.linspace(0,1,2000)
plt.plot(x, x, color="dimgray", linestyle="--", lw=2)
plt.plot(np.sort(a), np.linspace(0, 1, len(a)), color="royalblue", lw=3)
plt.plot(np.sort(b), np.linspace(0, 1, len(b)), color="darkseagreen", lw=3)
plt.plot(np.sort(c), np.linspace(0, 1, len(c)), color="indianred", lw=3)
plt.grid()
plt.xlim(-0.01,1.01)
plt.ylim(-0.01,1.01)
plt.legend(["CDF of Uniform Law", "CDF of $T$ under the null",\
            "CDF of $T$ under $\log(N)$ alt.", "CDF of $T$ under $\sqrt{N}$ alt."])
plt.title ("Emp. CDFs with $f_c=$%s over %s iterations" %(sample_size, iterations))
plt.tight_layout()
plt.savefig('fig2.png', format='png', dpi=300)
plt.show()