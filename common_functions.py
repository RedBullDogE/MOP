def dispersion(aver_val, array):
    """
    Function for calculating dispersion of given array

    :param aver_val: average value of array
    :param array: array for which the dispersion is calculated
    :return: dispersion value
    """
    squared_difference = [((array[i] - aver_val) ** 2) for i in range(len(array))]
    disp_val = sum(squared_difference) / len(array)
    return disp_val


def response_function(coeffs, variables):
    """
    Calculating value of response function according to regression equation:
        Y = a0 + a1 * x1 + a2 * x2 + a3 * x3 + ... + aN * xN

    :param coeffs: list of coefficients - ai
    :param variables: list of variables - xi
    :return: value of response function
    """
    y = coeffs[0] + sum([b * x for b, x in zip(coeffs[1:], variables)])
    return y
