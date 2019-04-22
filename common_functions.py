from functools import partial
from scipy.stats import f, t


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


def cohren(f1, f2, q=0.05):
    fisher_value = f.ppf(q=1 - (q / f2), dfn=f1, dfd=(f2 - 1) * f1)
    return fisher_value / (fisher_value + f2 - 1)


def student(f3):
    return partial(t.ppf, q=1 - 0.025)(df=f3)


def fisher(f3, f4):
    return partial(f.ppf, q=1 - 0.05)(dfd=f3, dfn=f4)


def beta_value(y_av_arr, norm_plan_point):
    """
    Calculating beta-value:
        Bj = 1 / N * sum(Yi, Xij), i = (1 -> K)
        where:
            N - number of plan points;
            K - number of factors;
            Yi - average value of Y for i row;
            Xij - normalized value of X for i row j column;
    :param y_av_arr: array of y average values
    :param norm_plan_point:
    :return: beta value for specified column
    """
    beta = sum(map(lambda y, x: y * x, y_av_arr, norm_plan_point)) / len(y_av_arr)
    return beta
