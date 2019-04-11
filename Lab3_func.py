from numpy.linalg import det


def calc_coeffs(nat_matrix, y_av_list):
    """
    Calculating normalized coefficients for regression equation,
    based on normalized matrix of X and list of average values of Y

    :param nat_matrix: normalized matrix of X
    :param y_av_list: average values of Y for appropriate plan points
    :return: list of normalized coefficients
    """
    x1_list = list(zip(*nat_matrix))[0]
    x2_list = list(zip(*nat_matrix))[1]
    x3_list = list(zip(*nat_matrix))[2]
    n = len(nat_matrix)
    mx1 = sum(x1_list) / n
    mx2 = sum(x2_list) / n
    mx3 = sum(x3_list) / n
    my = sum(y_av_list) / n

    a1 = sum(map(lambda x_1, y: x_1 * y, x1_list, y_av_list)) / n
    a2 = sum(map(lambda x_2, y: x_2 * y, x2_list, y_av_list)) / n
    a3 = sum(map(lambda x_3, y: x_3 * y, x3_list, y_av_list)) / n

    a11 = sum(map(lambda x: x ** 2, x1_list)) / n
    a22 = sum(map(lambda x: x ** 2, x2_list)) / n
    a33 = sum(map(lambda x: x ** 2, x3_list)) / n

    a12 = sum(map(lambda x_1, x_2: x_1 * x_2, x1_list, x2_list)) / n
    a13 = sum(map(lambda x_1, x_3: x_1 * x_3, x1_list, x3_list)) / n
    a23 = sum(map(lambda x_2, x_3: x_2 * x_3, x2_list, x3_list)) / n

    matrix0 = [
        [1, mx1, mx2, mx3],
        [mx1, a11, a12, a13],
        [mx2, a12, a22, a23],
        [mx3, a13, a23, a33]
    ]

    matrix1 = [
        [my, mx1, mx2, mx3],
        [a1, a11, a12, a13],
        [a2, a12, a22, a23],
        [a3, a13, a23, a33]
    ]

    matrix2 = [
        [1, my, mx2, mx3],
        [mx1, a1, a12, a13],
        [mx2, a2, a22, a23],
        [mx3, a3, a23, a33]
    ]

    matrix3 = [
        [1, mx1, my, mx3],
        [mx1, a11, a1, a13],
        [mx2, a12, a2, a23],
        [mx3, a13, a2, a33]
    ]

    matrix4 = [
        [1, mx1, mx2, my],
        [mx1, a11, a12, a1],
        [mx2, a12, a22, a2],
        [mx3, a13, a23, a3]
    ]

    b0 = det(matrix1) / det(matrix0)
    b1 = det(matrix2) / det(matrix0)
    b2 = det(matrix3) / det(matrix0)
    b3 = det(matrix4) / det(matrix0)

    return [b0, b1, b2, b3]


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
