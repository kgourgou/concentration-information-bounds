# Goal oriented bounds and helper functions

from scipy.optimize import minimize
from scipy import log,exp, sqrt, zeros, mean, isnan

from source.checker import checker

from warnings import warn

## Helper functions

def con_opt(objfun, x0, opt_tol=1e-6, maximize=False, bnds=((0,None),),method_name="L-BFGS-B"):
    '''constrained optimization of objective function
    with initial point x0 and tolerance opt_tol.

    By default, this function optimizes over c>0.

    >>> con_opt(lambda x: x**2, 0.1)
    0.0
    >>> con_opt(lambda x: exp(-x),0.1, maximize=True)
    1.0

    An example of constrained optimization.
    >>> con_opt(lambda x: x**2, 0.1, bnds=((1,3),))
    1.0

    And using a different than the standard method. Check docs for minimize.
    >>> con_opt(lambda x: x**2, 0.1, bnds=((1,3),) ,method_name="TNC")
    1.0
    '''

    if maximize == False:
        result = minimize(objfun,x0,bounds=bnds,tol=opt_tol,method=method_name)
        opt_val = result["fun"]
    else:
        ob = lambda c: -objfun(c) # reverse the function (for maximization)
        result = minimize(ob, x0, bounds=bnds,tol=opt_tol,method=method_name)
        opt_val = -result["fun"] # recover the correct value

    if result["success"]==False:
        print result["message"]

    return opt_val[0]


def mult_bounds(go_bound, kl_array, lcum, bounds=None):
    '''
      Computes the go_bounds multiple times, once
      for each value in the kl_array using the lcum
      log-cumulant.

    >>> lcum = lambda c: log((0.2+0.8*exp(c))*exp(-0.8*c))
    >>> klvals = [0.1,0.1,0.1,0.1]
    >>> bounds = mult_bounds(godiv, klvals, lcum)

    TODO This can also be done with the map function. What
    is more efficient?
    '''

    n = len(kl_array)
    bounds = zeros([2,n])
    for i in xrange(n):
        bounds[:,i] = go_bound(kl_array[i],lcum, bounds)

    return bounds


def logcum(c,data, center_data=None):
    '''
    Approximating the log-cumulant of sampled data.

    If center_data is a numerical value, the log-cumulant
    is computed by using that value to center the data. Otherwise
    the numerical average is used.

    >>> logcum(0,[0,1,0,1])
    0.0
    >>> logcum(0,[0,1,0,1],center_data=[0,1,2])
    0.0
    '''

    if center_data is None:
        return log(mean(exp(c*(data-mean(data)))))
    else:
        return log(mean(exp(c*(data-mean(center_data)))))



## The two general divergences
def gendiv(fdiv, lapl , opt_tol=1e-5, bounds=((0,None),),
           to_check=True,x0=None):
    '''  general goal-oriented divergence.

     Given an f-divergence, D_f(Q|P), we can write the inequality

     E_{P}[f]-E_{Q}[f]\leq
     inf_{c>0}\left \{ E_Q[f^*(c\hat{g})]/c+D_f(P\|Q)/c\right\}

     where f^* stands for the Laplace transform of f and
     \hat{g}=g-E_Q[g].

    Arguments
    =========
    fdiv : Value of the f-divergence.
    lapl : function handle. Should evaluate to E_Q[f^*(c\hat{g})].
    Is assumed to be a function of a single argument.
    opt_tol: tolerance to be passed to optimization.
    bnds: bounds used to constrain the optimization.
    to_check: Whether to check inconsistencies in the bounds.
    x0: initial condition to be passed to the optimizer. If not used,
    the klval is passed.

    Returns
    =======
    Upper and lower bounds for the bias.

    Example
    =======
    >>> fdiv = 0.1
    >>> lcum = lambda c: log(((0.2)+0.8*exp(c))*exp(-0.8*c))
    >>> lower,upper = godiv(fdiv, lcum)
    >>> (upper>0 and lower<0)
    True
    '''

    if x0==None:
        # upper bound
        objf = lambda c: (lapl(c)+fdiv)/c
        upper = con_opt(objf,fdiv,maximize=False, bnds=bounds)

        # lower bound
        objf = lambda c: (-lapl(-c)-fdiv)/c
        lower = con_opt(objf,fdiv, maximize=True,bnds=bounds)
    else:
        # upper bound
        objf = lambda c: (lapl(c)+fdiv)/c
        upper = con_opt(objf,x0,maximize=False, bnds=bounds,method="Nelder-Mead")

        # lower bound
        objf = lambda c: (-lapl(-c)-fdiv)/c
        lower = con_opt(objf,x0, maximize=True,bnds=bounds, method="Nelder-Mead")

    if to_check:
        checker(lower,upper)

    return lower, upper



## The two goal-oriented divergences

def godiv(klval, lcum, opt_tol=1e-5, bounds=((0,None),), to_check=True,x0=None):
    '''  goal-oriented divergence

    Arguments
    =========
    klval : Value of the Kullback-Leibler divergence.
    lcum : function that corresponds to a log-cumulant.
    Is assumed to be a function of a single argument.
    opt_tol: tolerance to be passed to optimization.
    bnds: bounds used to constrain the optimization.
    to_check: Whether to check inconsistencies in the bounds.
    x0: initial condition to be passed to the optimizer. If not used,
    the klval is passed.

    Example
    =======
    >>> klval = 0.1
    >>> lcum = lambda c: log(((0.2)+0.8*exp(c))*exp(-0.8*c))
    >>> lower,upper = godiv(klval, lcum)
    >>> (upper>0 and lower<0)
    True
    '''

    if x0==None:
        # upper bound
        objf = lambda c: (lcum(c)+klval)/c
        upper = con_opt(objf,klval,maximize=False, bnds=bounds)

        # lower bound
        objf = lambda c: (-lcum(-c)-klval)/c
        lower = con_opt(objf,klval, maximize=True,bnds=bounds)
    else:
        # upper bound
        objf = lambda c: (lcum(c)+klval)/c
        upper = con_opt(objf,x0,maximize=False, bnds=bounds)

        # lower bound
        objf = lambda c: (-lcum(-c)-klval)/c
        lower = con_opt(objf,x0, maximize=True,bnds=bounds)

    if to_check:
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

def hConvex(klval, a,b, mu=0, opt_tol=1e-5, mthd=None,x0=None):
    '''Hoeffding convex bounds for the bias

    Arguments::
    klval: value of KL
    a,b : upper and lower bound for the centralized X random variable.
    mu : expected value of x

    Example ::
    >>> hConvex(0.0,-1.0,1.0)
    (-0.0,0.0)
    '''

    if a>0 or b<0 or a>b:
        raise ValueError("a and b are the bounds of a centralized random variable.")


    # Hoeffding convex bound for the cumulant
    lcum_bound = lambda c: log(1/(b-a)*((b-mu)*exp(c*(a-mu))-(a-mu)*exp(c*(b-mu))))

    if x0 == None:
        objf = lambda c: (lcum_bound(c)+klval)/c
        upper = con_opt(objf, klval, method_name=mthd)

        objf = lambda c: (-lcum_bound(-c)-klval)/c
        lower= con_opt(objf,klval,maximize=True, method_name=mthd)
    else:
        objf = lambda c: (lcum_bound(c)+klval)/c
        upper = con_opt(objf, x0, method_name=mthd,)

        objf = lambda c: (-lcum_bound(-c)-klval)/c
        lower= con_opt(objf,x0,maximize=True, method_name=mthd)

    checker(lower,upper)

    return lower, upper


def Bennet(klval, b,sib,mu, opt_tol=1e-5, mthd=None,x0=None):
    '''Bennet convex bounds for the bias

    Arguments::
    klval: value of KL
    b: upper bound of random variable
    sib: upper bound for variance
    mu: mean of random variable

    Example ::
    TODO Write a docstring example
    '''

    # Bennet convex bound for the cumulant
    b_cent_sq = (b-mu)**2

    lcum_bound = lambda c: (log(1/(b_cent_sq+sib))
    +log(b_cent_sq*exp(-c*sib/(b-mu))+sib*exp(c*(b-mu))))

    if x0 == None:
        objf = lambda c: (lcum_bound(c)+klval)/c
        upper = con_opt(objf, klval, method_name=mthd)

    else:
        objf = lambda c: (lcum_bound(c)+klval)/c
        upper = con_opt(objf, x0, method_name=mthd,)


    lower = 0 # bennet doesn't provide lower bounds
    checker(lower,upper)

    return lower, upper




if __name__ == "__main__":
    import doctest
    doctest.testmod()
