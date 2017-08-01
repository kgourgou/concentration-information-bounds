"""
  Running some additional tests on the KL sampling.
"""

import scipy as sp
import matplotlib.pyplot as pl
import seaborn

from kldivs import KLSampling, KLNormal

## Example of two Normally distributed random variables

# ratio of the two pdfs
ratio = lambda x: sp.exp(-(x-1)**2/2.0)/sp.exp(-x**2.0/2.0)

print "KL = ", KLNormal(1,1,0,1)

N = 1000
M = 100

kl  = sp.zeros(M)

# histogram
for i in range(M):
    data = sp.randn(N)
    kl[i]= KLSampling(ratio,data)

pl.hist(kl)
pl.title("KL estimates")
pl.show()

