from numpy.linalg import det


def f_coeffs(array):
    """
    Calculating Fuv coefficients for array.
        if disp(u) >= disp(v): Fuv = disp(Yu) / disp(Yv)
        else: Fuv = disp(Yv) / disp(Yu)

    :param array: array for which the Fuv coefficients are calculated
    :return: list of all Fuv coefficients
    """
    f_uv_list = list()
    for i in range(len(array) - 1):
        for j in range(i + 1, len(array)):
            if i != j:
                f_uv_list.append(max(array[i], array[j]) / min(array[i], array[j]))

    return f_uv_list


def calc_norm_coeff(norm_matrix, y_av_list):
    """
    Calculating normalized coefficients for regression equation,
    based on normalized matrix of X and list of average values of Y

    :param norm_matrix: normalized matrix of X
    :param y_av_list: average values of Y for appropriate plan points
    :return: list of normalized coefficients
    """
    x1_list = list(zip(*norm_matrix))[0]
    x2_list = list(zip(*norm_matrix))[1]
    n = len(norm_matrix)
    mx1 = sum(x1_list) / n
    mx2 = sum(x2_list) / n
    my = sum(y_av_list) / n
    a1 = sum(map(lambda x: x ** 2, x1_list)) / n
    a2 = sum(map(lambda x_1, x_2: x_1 * x_2, x1_list, x2_list)) / n
    a3 = sum(map(lambda x: x ** 2, x2_list)) / n
    a11 = sum(map(lambda x_1, y: x_1 * y, x1_list, y_av_list)) / n
    a22 = sum(map(lambda x_2, y: x_2 * y, x2_list, y_av_list)) / n

    matrix0 = [[1, mx1, mx2],
               [mx1, a1, a2],
               [mx2, a2, a3]]
    matrix1 = [[my, mx1, mx2],
               [a11, a1, a2],
               [a22, a2, a3]]
    matrix2 = [[1, my, mx2],
               [mx1, a11, a2],
               [mx2, a22, a3]]
    matrix3 = [[1, mx1, my],
               [mx1, a1, a11],
               [mx2, a2, a22]]

    b0 = det(matrix1) / det(matrix0)
    b1 = det(matrix2) / det(matrix0)
    b2 = det(matrix3) / det(matrix0)

    return [b0, b1, b2]


def calc_nat_coeff(x1_min, x1_max, x2_min, x2_max, norm_coeffs):
    """
    Calculating naturalized coefficients for regression equation,
    based on plan_matrix (minimal and maximal values of each factor) and
    normalized coefficients

    :param x1_min: minimal value of first factor
    :param x1_max: maximal value of first factor
    :param x2_min: minimal value of second factor
    :param x2_max: maximal value of second factor
    :param norm_coeffs: list of normalized coefficients
    :return: list of naturalized coefficients
    """
    dx1 = (x1_max - x1_min) / 2
    dx2 = (x2_max - x2_min) / 2
    x10 = (x1_max + x1_min) / 2
    x20 = (x2_max + x2_min) / 2
    a0 = norm_coeffs[0] - norm_coeffs[1] * x10 / dx1 - norm_coeffs[2] * x20 / dx2
    a1 = norm_coeffs[1] / dx1
    a2 = norm_coeffs[2] / dx2

    return [a0, a1, a2]
