"""
  Moment generating functions.
"""

from scipy import log, exp, sqrt
from scipy.stats import norm

def MGFtruncatedN(t,mu,ssq,alpha=-0.5,beta=0.5):
    # Moment Generating Function of the truncated Normal.
    # a, b are the bounds of the normal distribution.

    ss = sqrt(ssq)

    a = (alpha-mu)/ss
    b = (beta-mu)/ss

    numerator = norm.cdf(b-ss*t)-norm.cdf(a-ss*t)
    denominator= norm.cdf(b) - norm.cdf(a)
    return exp(ssq*(t**2.0)/2.0+mu*t) * numerator/denominator



def MGFNormal(c,mu=0,sigma=1):
    """
    MGF of the normal distribution.

    mu: mean
    sigma: standard error

    >>> MGFNormal(0,mu=2,sigma=5)
    1.0
    """
    if sigma<0:
        raise ValueError("sigma cannot be negative.")

    return exp(mu*c+sigma**2*c**2*0.5)



def MGFBernoulli(c,p):
    """
    Value of the MGF of the Bernoulli dist. at value t
    when the probability of success is p.

    >>> MGFBernoulli(0,0.3)
    1.0
    """
    if p<0 or p>1:
        raise ValueError("0<=p<=1 is violated.")

    return 1-p+p*exp(c)


def MGFGamma(c, k, theta):
    '''
    shape-scale parameterization of the mgf
    '''
    t = c*theta
    if t<1:
        return 1/(1-t)**k
    else:
        return 1e+8


if __name__ == "__main__":
    import doctest
    doctest.testmod()
