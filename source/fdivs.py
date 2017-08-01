"""
Some f-divergences. Remember that f is a convex function.
"""

from scipy import log
from scipy.special import gamma, digamma
import scipy

def chisquaredNormal(mq,sq,mp,sp):
    """
    Chi-squared divergence for a pair of normals

    Arguments
    =========
    mq,mp: mean of q and p
    sq, sp: standard deviation of q and p


    Examples:
    ========
    >>> mp=1; mq=1; sq=1; sp=1;
    >>> chisquaredNormal(mp,mq,sq,sp)
    0.0

    >>> mp=0; mq=1; sq=1; sp=1;
    >>> val = chisquaredNormal(mp,mq,sq,sp)
    >>> abs(scipy.exp(1)-1-val)<1e-10
    True
    """
    if sq<=0 or sp<=0:
        raise ValueError("Standard devs. have to be positive.")

    if sq*scipy.sqrt(2)<sp:
        raise ValueError("chi^2 divergence cannot handle this case yet.")

    mbias = mq-mp
    sbias = 2*sq**2-sp**2
    value = scipy.exp(mbias**2/sbias)*sq**2/scipy.sqrt(sbias)-1

    return value



if __name__ == "__main__":
    import doctest
    doctest.testmod()
