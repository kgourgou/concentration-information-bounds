import scipy
from scipy.optimize import minimize


class UBound(object):
    """
    Class to use when defining new UQ-bounds.

    Attributes:
    mgf: MGF or bound to the MGF.
    Assumed to be a function with arguments (c,data)

    data: Dictionary with the data required to implement
    the bound.

    lcum: log-cumulant or bound to the log-cumulant. 
    """

    def __init__(self, data, mgf=None,lcum=None):

        if mgf==None and lcum==None:
            raise IOError("Either mgf or lcum need to be passed as input.")
        elif mgf!=None:
            self.mgf = mgf
            self.lcum = lambda c: scipy.log(mgf(c,data))
        else:
            self.lcum = lambda c : lcum(c,data)

        self.data = data

    # TODO support for explicit bounds

    def eval(self,eta,x0=None,obounds=(0,None), opt_tol=1e-5, to_check=True):

        """
        Evaluates the U-bound through optimization.

        Arguments:
        eta : Value of the Kullback-Leibler divergence.
        opt_tol: tolerance to be passed to optimization.
        obounds: bounds used to constrain the optimization.
        x0: initial condition to be passed to the optimizer. If not used,
        the square root of the value of the KL will be used.
        This is motivated by the linearized bounds
        and should work well for small values of the KL.
        """

        # No optimization required for very small
        # values of eta
        if eta<1e-8:
            return 0,0

        if x0==None:
            x0 = scipy.sqrt(eta)

        # # upper bound
        objf = lambda c: (self.lcum(c)+eta)/c
        upper = self.__con_opt(objf,x0,maximize=False, bnds=obounds)
        # lower bound
        objf = lambda c: (-self.lcum(-c)-eta)/c
        lower = self.__con_opt(objf,x0, maximize=True,bnds=obounds)


        if to_check:
            # Simple sanity checks on the bounds
            self.__checker(lower,upper)

        return lower, upper


    #################################### Helper functions

    def __con_opt(self, objfun, x0, opt_tol=1e-6, maximize=False, bnds=(0,None),
                method_name="L-BFGS-B"):
        '''constrained optimization of objective function
        with initial point x0 and tolerance opt_tol.

        By default, this function optimizes over c>0.

        >>> __con_opt(lambda x: x**2, 0.1)
        0.0
        >>> __con_opt(lambda x: exp(-x),0.1, maximize=True)
        1.0

        An example of constrained optimization.
        >>> __con_opt(lambda x: x**2, 0.1, bnds=((1,3),))
        1.0

        And using a different than the standard method. Check docs for minimize.
        >>> __con_opt(lambda x: x**2, 0.1, bnds=((1,3),) ,method_name="TNC")
        1.0
        '''


        if maximize == False:
            result = minimize(objfun,x0,bounds=(bnds,),tol=opt_tol,method=method_name)
            opt_val = result["fun"]
        else:
            ob = lambda c: -objfun(c) # reverse the function (for maximization)
            result = minimize(ob, x0, bounds=(bnds,),tol=opt_tol,method=method_name)
            opt_val = -result["fun"] # recover the correct value

        if result["success"]==False:
            print result["message"]

        return opt_val[0]

    @staticmethod
    def __checker(lower,upper):
        """
        Some easy checks on the bounds.
        """
        if lower>upper:
            raise ValueError("Possible optimization error: lower bound > upper bound.")

        if lower>0:
            raise ValueError("Lower bound has to be non-positive.")

        if upper<0:
            raise ValueError("Upper bound has to be non-negative.")


if __name__=="__main__":
    import scipy as sc
    import ubounds

    # Two different ways of defining the Hoeffding bound
    mgf = lambda c,d: sc.exp(c**2/2.0*(d["b"]-d["a"])**2)
    lcum= lambda c,d: c**2/2.0*(d["b"]-d["a"])**2
    data = {"a":-1.0,"b":1.0}
    hoef = UBound(data,lcum=lcum)
    print(hoef.eval(0.1))
    hoef = UBound(data,mgf=mgf)
    print(hoef.eval(0.1))
