from source.checker import checker
from scipy import sqrt

from warnings import warn


## linearized bounds
def lbound(vp,klval):
    """
    linearized bound that approximates the GO divergence when
    the value of the KL divergence is small.

    Arguments::
    klval: the value of the KL divergence
    vp: the corresponding variance of the P distribution
    """

    if klval > 1.0:
        warn("KL value is too large, KL={}.".format(klval))
        warn("The approximation may not be accurate.")

    return sqrt(2*vp*klval)

def lbound_approx(vq, klval):
    """
    An approximation to the linearized bound
    of P by using information from the Q measure.

    Arguments::
    klval: the value of the KL divergence
    vq : the variance of Q
    """
    return sqrt(vq*klval*2)+1/(2*sqrt(vq))*(sqrt(vq+klval*vq*2)*sqrt(2*klval))

## Variants that depend on bounds of the random variable or
## assumptions on the tail behavior (sub-gaussianity).

def BerKontorovich(klval,a,b,mu):
    """
    Berend-Kontorovich exponential bound.

    This is a sharpened sub-gaussian that improves
    on Hoeffding by swapping out the 1/8 part for an
    improved constant.

    Arguments::
    klval: value of KL
    a,b: bounds of the un-centralized random variable
    mu: mean value of the un-centralized random variable

    Example::
    >>> BerKontorovich(0.,-1,1,0)
    (-0.0, 0.0)
    """

    if a>b:
        raise ValueError("a>b")

    d = b-a
    p = (mu-a)/d

    # compute Berend-Kontorovich constant
    if p==0:
        cbk=0
    elif p<0.5:
        cbk = (1-2.0*p)/(4.0*log((1-p)/p))
    else:
        cbk = p*(1-p)/2.0

    upper = 2*sqrt(cbk)*d*sqrt(klval)
    lower = 0 # this bound only works for c>0
    checker(lower, upper)
    return lower, upper


### Hoeffding


def hExp(klval, a,b, opt_tol=1e-5):
    '''Hoeffding exponential bound

    Arguments::
    klval: value of KL
    a,b : upper and lower bound for the X random variable.


    Example ::
    >>> hExp(0, -1,1)
    (-0.0, 0.0)
    >>> hExp(1,0,0)
    (-0.0, 0.0)
    '''

    if a>b:
        raise ValueError("a and b are the bounds of a centralized random variable.")

    # This bound has an explicit solution
    upper = sqrt(2)/2.0*(b-a)*sqrt(klval)
    lower = -upper

    checker(lower,upper)

    return lower, upper

def hExpVar(klval, variance_proxy):
    '''Hoeffding exponential bound with variance proxy.
    This is to be used with bounded random variables.

    Arguments::
    klval: value of KL
    variance_proxy: Either the true variance or an upper bound.

    Example ::
    >>> hExpVar(0,1)
    (-0.0, 0.0)
    >>> hExpVar(0,2)
    (-0.0, 0.0)
    '''

    if variance_proxy<0:
        raise ValueError("variance proxy has to be non-negative.")

    # This bound has an explicit solution
    upper = sqrt(2)*sqrt(variance_proxy*klval)
    lower = -upper

    checker(lower,upper)

    return lower, upper

## Variants that depend on bounds of the random variable or
## assumptions on the tail behavior (sub-gaussianity).

def BerKontorovich(klval,a,b,mu):
    """
    Berend-Kontorovich exponential bound.

    This is a sharpened sub-gaussian that improves
    on Hoeffding by swapping out the 1/8 part for an
    improved constant.

    Arguments::
    klval: value of KL
    a,b: bounds of the un-centralized random variable
    mu: mean value of the un-centralized random variable

    Example::
    >>> BerKontorovich(0.,-1,1,0)
    (-0.0, 0.0)
    """

    if a>b:
        raise ValueError("a>b")

    d = b-a
    p = (mu-a)/d

    # compute Berend-Kontorovich constant
    if p==0:
        cbk=0
    elif p<0.5:
        cbk = (1-2.0*p)/(4.0*log((1-p)/p))
    else:
        cbk = p*(1-p)/2.0

    upper = 2*sqrt(cbk)*d*sqrt(klval)
    lower = 0 # this bound only works for c>0
    checker(lower, upper)
    return lower, upper


### Hoeffding


def hExp(klval, a,b, opt_tol=1e-5):
    '''Hoeffding exponential bound

    Arguments::
    klval: value of KL
    a,b : upper and lower bound for the X random variable.


    Example ::
    >>> hExp(0, -1,1)
    (-0.0, 0.0)
    >>> hExp(1,0,0)
    (-0.0, 0.0)
    '''

    if a>b:
        raise ValueError("a and b are the bounds of a centralized random variable.")

    # This bound has an explicit solution
    upper = sqrt(2)/2.0*(b-a)*sqrt(klval)
    lower = -upper

    checker(lower,upper)

    return lower, upper

def hExpVar(klval, variance_proxy):
    '''Hoeffding exponential bound with variance proxy.
    This is to be used with unbounded random variables. 

    Arguments::
    klval: value of KL
    variance_proxy: Either the true variance or an upper bound.

    Example ::
    >>> hExpVar(0,1)
    (-0.0, 0.0)
    >>> hExpVar(0,2)
    (-0.0, 0.0)
    '''

    if variance_proxy<0:
        raise ValueError("variance proxy has to be non-negative.")

    # This bound has an explicit solution
    upper = sqrt(2)*sqrt(variance_proxy*klval)
    lower = -upper

    checker(lower,upper)

    return lower, upper
