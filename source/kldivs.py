# KL divergences

from scipy import log
from scipy.special import gamma, digamma
import scipy

# TODO missing KLTruncatedNormal

def KLSampling(ratio, data):
    """
    Computes KL(q||p) by using samples of q.

    Some very elementary checks are made to make sure the numbers
    returned make sense. The estimator used has the same properties
    (and issues) as when one is doing importance sampling, i.e.,
    problems with estimating the ratio of partition functions.

    Arguments::
     ratio: The ratio of the normalized (or un-normalized) pdfs or pmfs
     of the form p/q.

    data: Data to compute the estimator on.

    Returns::
     est: estimate of the KL divergence.

    Examples::
    The case of equal distributions.
    >>> from scipy import exp
    >>> rat = lambda x: exp(-(x)**2/2)/exp(-(x)**2/2)
    >>> data = scipy.randn(1000) # sample from normal with mean 1
    >>> est = KLSampling(rat, data)
    >>> est<1e-7
    True

    Different means, same deviations.
    >>> mu = 0.4
    >>> rat = lambda x: exp(-(x-mu)**2/2)/exp(-(x-1.0)**2/2)
    >>> data = scipy.randn(100000)+1
    >>> est = KLSampling(rat, data)
    >>> abs(est-KLNormal(mu,1.0,1,1.0))<1
    True


    Testing if mapping works well for distributions of more than one
    parameter.
    >>> rat = lambda x: exp(-(x[0]+x[1])/2**2)
    >>> data = scipy.randn(100,2)
    >>> KLSampling(rat,data)<0.6
    True
    """

    dim = len(data.shape)
    est_a = 0.0
    est_b = 0.0


    if dim == 1:
        n = len(data)
        est_a = sum([log(1.0/ratio(x)) for x in data])
        est_b = sum([ratio(x) for x in data])
        # for i in xrange(n):
        #     val = ratio(data[i])
        #     est_a = est_a+log(1.0/val)
        #     est_b = est_b+val

        if scipy.isnan(est_a):
            raise ValueError("est_a is nan")
        if scipy.isnan(est_b):
            raise ValueError("est_b is nan")

        est = est_a/n + log(est_b/n) # total estimate

    else:
        # nxm format assumed, where every row accounts for data
        # every column for variables
        n = scipy.size(data,0)
        for i in xrange(n):
            val = ratio(data[i,:])
            est_a = est_a+log(1.0/val)
            est_b = est_b+val

        if scipy.isnan(est_a):
            raise ValueError("est_a is nan")
        if scipy.isnan(est_b):
            raise ValueError("est_b is nan")

        est = est_a/n + log(est_b/n) # total estimate

    if est<0 or est_b<0:
        raise ValueError("Insufficient data to converge.")


    return est


def KLNormal(mu1,sigma1,mu2, sigma2):
    '''
    KL between two Normal distributions.
    >>> KLNormal(1,2,1,2)
    0.0
    '''
    if sigma1<=0 or sigma2<=0:
        raise ValueError("sigma1 and sigma2 have to be positive.")

    sigma1 = float(sigma1)
    mu1 = float(mu1)

    return log(sigma2/sigma1) + float(sigma1**2+(mu1-mu2)**2)/(2*sigma2**2)-0.5

def KLBernoulli(p1,p2):
    '''
    KL between two Bernoulli
    >>> KLBernoulli(0.3,0.3)
    0.0
    '''
    q1 = 1-p1
    q2 = 1-p2
    return p1*log(p1/p2) + q1*log(q1/q2) 

def KLGamma(k1, th1, k2, th2):
    '''
    KL between two Gamma distributions with
    shape-scale parameterizations.
    >>> KLGamma(1,2,1,2)
    0.0
    '''

    if k1<=0 or th1<=0 or k2<=0 or th2<=0:
        raise ValueError("parameters of gamma need to be positive.")

    partA = (th1-th2)/th2*k1
    partB = log(gamma(k2)*th2**k2/(gamma(k1)*th1**k1))
    partC = (k1-k2)*(log(th1)+digamma(k1))
    return partA+partB+partC

def KLExponential(l1, l2):
    '''
    KL between two exponential distributions.
    >>> KLExponential(1,1)
    0.0
    >>> abs(KLExponential(1,2)-(log(0.5)+1))<1e-8
    True
    '''

    if l1<=0 or l2<=0:
        raise ValueError("parameters of exponential need to be positive.")

    l1 = float(l1)
    l2 = float(l2)
    return log(l1/l2)+(l2-l1)/l1



if __name__ == "__main__":
    import doctest
    doctest.testmod()
