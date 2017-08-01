def checker(lower,upper):
    """
    Some easy checks on the bounds.
    """
    if lower>upper:
        raise ValueError("Possible optimization error: lower bound > upper bound.")

    if lower>0:
        raise ValueError("Lower bound has to be non-positive.")

    if upper<0:
        raise ValueError("Upper bound has to be non-negative.")
