"""

  MGF concentration bounds

    In most cases, the bounds apply either for a bounded
    random variable X in (a, b), or for a sub-Gaussian
    random variable with a known variance proxy. The Bennet
    bound requires a variable in (-inf, b) with a variance
    proxy.

Notation:

math:: \bar{X}=E[X]=\mu
math:: \hat{a}=a-\mu
math:: \hat{b}=b-\mu

"""

import scipy as sc

class mgf_bounds(object):

    def __init__(self,
                 bound,
                 data_for_bound,
                 name = None):
        self.bound = bound
        self.data = data_for_bound
        self.name = name or bound

    def eval(self, c):
        return bound(c, self.data)


def hoeffding(c, data):
    """
    Hoeffding sub-Gaussian bound

    ::math exp(c^2 (b-a)^2/8)

    or

    ::math exp(c^2 sb^2/2)

    Works for either bounded random variables in (a,b)
    or sub-Gaussian random variables.

    Arguments:
    c: float. 
    data: dict, must contain either "a" and "b" or
    "sb".

    >>> data = {"a":-1, "b":1}
    >>> hoeffding(0, data) == 1.0
    True

    >>> data = {"sb":1.0}
    >>> hoeffding(0, data) == 1.0
    True
   
    """

    __checker(data)

    if "a" in data\
       and "b" in data:
        sb = data["b"]-data["a"]

    elif "sb" in data:
        sb = sc.sqrt(4)*data["sb"]

    else:
        raise KeyError("Requires either (a,b) (bounded random variable)" +
                       "or sb.")

    return sc.exp(c*sb**2/8)


def berend_kontorovich(c, data):
    """
    Berend-Kontorovich exponential bound.

    This is a sharpened version of the Hoeffding bound for
    bounded random variables that swaps out 1/8 for an
    improved constant.

    Arguments:
    c: float
    data: dict, must contain the bounds of the random variable, a, b, and
    the expected value of X, mu. If mu is not included, it is assumed that
    X is already centralized. 
    """

    __checker(data)
    if "mu" in data:
        mu = data["mu"]
    else:
        mu = 0

    if c<0:
        raise ValueError("This bound only works for c>=0.")

    if "a" in data and "b" in data:
        bhat  = data["b"] - mu
        ahat  = data["a"] - mu
        diff = bhat - ahat
        p = -ahat/diff

        if p == 0:
            cbk=0
        elif p<0.5:
            cbk = (1-2.0*p)/(4.0*log((1-p)/p))
        else:
            cbk = p*(1-p)/2.0

    # TODO finish this
    return NotImplemented

def bennet_ab(c, data):
    """
    Bound for the MGF of a bounded random variable X in (a, b).

    ::math 
    ::math \hat{b}=b-
    ::math E[e^{c(X-\bar{X})}] \leq  \hat{b}/(b-a)\exp(c\cdot \hat{a})
    -\hat{a}/(b-a)\exp(c \cdot \hat{b})

    >>> data = {"a":-1, "b":2}
    >>> bennet_ab(0, data) == 1.0
    True

    >>> data = {"a":-1, "b":2, "mu":1}
    >>> bennet_ab(0, data) == 1.0
    True

    """
    __checker(data)
    if "mu" in data:
        mu = data["mu"]
    else:
        # assumed that bounds are already centralized
        mu = 0
    
    bhat = data["b"]-mu
    ahat = data["a"]-mu
    diff = bhat-ahat

    result = bhat/diff*sc.exp(c*ahat)-ahat/diff*sc.exp(c*bhat)
    return result


def bennet(c, data):
    """
    Bound for the MGF of an upper bounded random variable,

    ::math X<=b

    with bounded variance

    ::math var[X] <= sb^2

    Arguments:
    c: float
    data: dict, contains b, sb and mu. If mu is not provided,
    it is assumed that X is a centralized random variable.

    >>> bennet(0, {"b":1, "sb":2}) == 1.0
    True
    """

    __checker(data)

    if c<0:
        raise ValueError("This bound only works for c>=0.")

    if "mu" in data:
        mu = data["mu"]
    else:
        # assumed that bounds are already centralized
        mu = 0

    if "b" in data and "sb" in data:
        bhat = float(data["b"]-mu)
        bhat_sq = bhat**2
        sb_sq = float(data["sb"]**2)
        sum_of_sq = bhat_sq + sb_sq

        part1 = bhat_sq/sum_of_sq*sc.exp(-c*sb_sq/bhat)
        part2 = sb_sq/sum_of_sq*sc.exp(c*bhat)
    else:
        raise KeyError("Missing data. data = {}".format(data))

    return part1+part2



def __checker(data):
    """
    Check bound data for inconsistencies.
    """

    if "b" in data and "a" in data:
        if data["b"] < data["a"]:
            raise ValueError("a should be smaller than b")

    if "b" in data and "mu" in data:
        b = data["b"]-data["mu"]
        if b<=0:
            raise ValueError("b-mu={} <=0".format(b))

        if "a" in data:
            a = data["a"]-data["mu"]
            if a>=0:
                raise ValueError("a-mu={} >=0.".format(a))
    elif "b" in data:
        b = data["b"]
        if b<=0:
            raise ValueError("b = {} <= 0".format(b))

    if "sb" in data:
        if data["sb"] <= 0:
            raise ValueError("sb = {} <= 0".format(sb))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
