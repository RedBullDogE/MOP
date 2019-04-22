from itertools import combinations, product
from prettytable import PrettyTable
from random import uniform
import numpy as np
from common_functions import dispersion, cohren, student, fisher, beta_value

x1_min, x1_max = -2, 4
x2_min, x2_max = -10, 8
x3_min, x3_max = -3, 6
x_min, x_max = (x1_min + x2_min + x3_min) / 3, (x1_max + x2_max + x3_max) / 3
y_min = 200 + x_min
y_max = 200 + x_max
x_min_list = [x1_min, x2_min, x3_min]
x_max_list = [x1_max, x2_max, x3_max]
x0_list = [(x_max_list[i] + x_min_list[i]) / 2 for i in range(len(x_max_list))]


class Coefficient:
    def __init__(self, number):
        self.number = number
        self.value = 0

    def set_value(self, value):
        self.value = float(value)

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return self.value * other

    def __truediv__(self, other):
        return self.value * other

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def val(self):
        return self.value

    def text_present(self):
        return "b{}".format(self.number)

    def __str__(self):
        return "{}".format(self.value)

    __repr__ = __str__


class Feature(Coefficient):
    def __init__(self, number, val=0):
        super().__init__(number)
        self.number = number
        self.value = val

    def val(self):
        return self.value

    def __str__(self):
        return "{}".format(self.value)

    __repr__ = __str__


class Regression:
    def __init__(self, n, interaction=0, quadratic=False):
        self.__coeff_list = []
        self.__feature_list = []
        self.__terms = []
        self.__quadratic = quadratic

        if interaction == 2 ** n - n - 1 and quadratic:
            coeff_list_size = 2 ** n + 3
        elif interaction == 2 ** n - n - 1 and not quadratic:
            coeff_list_size = 2 ** n
        else:
            try:
                interaction = int(interaction)
                if interaction > 2 ** n - n - 1:
                    raise ValueError("interaction must be less than 2 ** n - n")
                coeff_list_size = n + interaction + 1
            except TypeError:
                raise TypeError("interaction must be str(int) or 'max'")

        feature_list_size = n
        self.__feature_list = [Feature(i + 1) for i in range(n)]
        self.__coeff_list = [Coefficient(i) for i in range(coeff_list_size)]
        for i in range(min(feature_list_size, interaction + 1)):
            self.__terms.extend(list(combinations(self.__feature_list, i + 1)))

        self.__terms = self.__terms[:n + interaction]

        if interaction == 2 ** n - n - 1 and quadratic:
            self.__terms.extend([(x, x) for x in self.__feature_list])

    def resp_func_val(self, coeff_list=None, x_list=None):
        def resp_func(coeffs, terms):
            term_value_list = [1] + [mul(term) for term in terms]
            y = sum(map(lambda b, x: b * x, coeffs, term_value_list))
            return y

        if coeff_list is None and x_list is not None:
            self.set_features(x_list)
        elif x_list is None and coeff_list is not None:
            self.set_coeffs(coeff_list)
        elif x_list is not None and coeff_list is not None:
            self.set_features(x_list)
            self.set_coeffs(coeff_list)

        return resp_func(self.__coeff_list, self.__terms)

    def set_features(self, x_list):
        if self.num_features == len(x_list):
            for i in range(self.num_features):
                self.__feature_list[i].set_value(x_list[i])
        else:
            raise IndexError("length of input feature list must be equal to the size of initial feature list")

    def clean_features(self):
        self.__feature_list = [Feature(i + 1) for i in range(self.num_features)]

    def set_coeffs(self, coeff_list):
        if self.num_coeffs == len(coeff_list):
            for i in range(self.num_coeffs):
                self.__coeff_list[i].set_value(coeff_list[i])
        else:
            raise IndexError("length of input coefficient list must be equal to the size of initial coefficient list")

    def clean_coeffs(self):
        self.__feature_list = [Coefficient(i) for i in range(self.num_coeffs)]

    @property
    def terms_list(self):
        return [tuple(x.val() for x in term) for term in self.__terms]

    @property
    def coeffs_list(self):
        return self.__coeff_list

    @property
    def num_features(self):
        return len(self.__feature_list)

    @property
    def num_coeffs(self):
        return len(self.__coeff_list)

    def __str__(self):
        string_reg = "{}".format(self.__coeff_list[0])
        for i in range(len(self.__terms)):
            row = " + {}" + "*{}" * len(self.__terms[i])
            string_reg += row.format(self.__coeff_list[i + 1], *self.__terms[i])

        return string_reg


def mul(arr):
    res = 1
    for el in arr:
        res *= el
    return round(res, 3)


def naturalize(min_val, max_val, val):
    x0 = (max_val + min_val) / 2
    dx = max_val - x0
    return val * dx + x0


def calc_coeffs(x_matrix, y_list):
    features = list(zip(*x_matrix))
    mx_list = [sum(features[i]) / len(features[i]) for i in range(len(features))]
    a_ij_matrix = [
        [
            sum([x1 * x2 for x1, x2 in zip(features[i], features[j])]) / len(features[i])
            for j in range(len(features))
        ]
        for i in range(len(features))
    ]
    a_i_matrix = [sum([x * y for x, y in zip(features[i], y_list)]) / len(features[i]) for i in range(len(features))]
    my = sum(y_list) / len(y_list)

    res_matrix = [
        [1, *mx_list],
        *[[mx_list[j], *a_ij_matrix[j]] for j in range(len(a_ij_matrix))]
    ]
    b = list(np.linalg.solve(res_matrix, [my, *a_i_matrix]))

    return b


def gen_norm_matrix(f_num, n=0, quadratic=False, y_list=None):
    k = 2 ** f_num - n
    test_reg = Regression(f_num, 2 ** f_num - f_num - n - 1, quadratic)
    x_variations = [list(item) for item in product([-1, 1], repeat=f_num)]
    matrix = []
    for i in range(k):
        test_reg.set_features(x_variations[i])
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    features = [
        [-1.215, 0, 0],
        [1.215, 0, 0],
        [0, -1.215, 0],
        [0, 1.215, 0],
        [0, 0, -1.215],
        [0, 0, 1.215],
        [0, 0, 0]
    ]

    for i in range(len(features)):
        test_reg.set_features(features[i])
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    if y_list is not None:
        for i in range(len(matrix)):
            matrix[i].append(y_list[i])
    return matrix


def gen_nat_matrix(norm_matr):
    k = 8
    f_num = 3
    test_reg = Regression(f_num, 2 ** f_num - f_num - 1, True)
    x_variations = [list(item) for item in product([-1, 1], repeat=f_num)]
    x_variations_nat = [[x_min_list[j] if x_variations[i][j] < 0 else x_max_list[j] for j in range(3)] for i in
                        range(len(x_variations))]
    matrix = []
    for i in range(k):
        test_reg.set_features(x_variations_nat[i])
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    features = [
        [-1.215, 0, 0],
        [1.215, 0, 0],
        [0, -1.215, 0],
        [0, 1.215, 0],
        [0, 0, -1.215],
        [0, 0, 1.215],
        [0, 0, 0]
    ]
    nat_features = [[naturalize(x_min_list[i], x_max_list[i], plan_point[i]) for i in range(len(plan_point))] for
                    plan_point in features]
    for plan_point in nat_features:
        test_reg.set_features(plan_point)
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    return matrix


def make_pretty_table(matr):
    table = PrettyTable()
    header = ["x1", "x2", "x3", "x1*x2", "x1*x3", "x2*x3", "x1*x2*x3", "x1 ^ 2", "x2 ^ 2", "x3 ^ 2", "Yav"]

    table.field_names = header[:len(matr[0])]
    for row in matr:
        table.add_row(row)
    return table


l = 1.215  # l - shoulder length
m = 3  # number of experiments
k = 3  # number of features
N = 2 ** k + 2 * k + 1  # number of plan points
norm_matrix = gen_norm_matrix(3, 0, True)
nat_matrix = gen_nat_matrix(norm_matrix)

while True:
    print("m = {}".format(m))
    y_values = [[uniform(y_min, y_max) for i in range(m)] for j in range(len(norm_matrix))]
    y_average_values = [round(sum(yi) / len(yi), 3) for yi in y_values]

    norm_matrix_table = make_pretty_table(gen_norm_matrix(3, 0, True, y_average_values))
    nat_matrix_table = make_pretty_table(nat_matrix)

    # -----------------------

    b_list = calc_coeffs(nat_matrix, y_average_values)
    test_regression = Regression(3, 4, True)
    test_regression.set_coeffs(b_list)
    new_y_values_correct = [test_regression.resp_func_val(x_list=nat_matrix[i][:3]) for i in range(len(nat_matrix))]
    y_dispersion_list = [dispersion(y_average_values[i], y_values[i]) for i in range(len(y_values))]

    # ------------ Cochren's criteria ------------
    gp = max(y_dispersion_list) / sum(y_dispersion_list)
    f1, f2 = m - 1, 2 ** k + 2 * k + 1
    if gp < cohren(f1, f2):
        break
    m += 1
    print("Dispersion is heterogeneous. Incrementing m...")

# ------------ Student's criteria ------------
f3 = f1 * f2
t_t = student(f3)

num_of_exp = 10
disp_b = sum(y_dispersion_list) / num_of_exp
disp_bs = disp_b / (num_of_exp * m)
std_bs = disp_bs ** (1 / 2)
beta_list = [beta_value(y_average_values, list(zip(*norm_matrix))[i]) for i in range(num_of_exp)]
t_list = [abs(beta_list[i]) / std_bs for i in range(num_of_exp)]
new_nat_coeffs = [b_list[i] if i < 1 or i > 7 else 0.0 for i in range(11)]

# ------------- Fisher's criteria -------------
d = len(new_nat_coeffs) - len([coeff for coeff in new_nat_coeffs if coeff != 0])
f4 = N - d
f_t = fisher(f3, f4)

new_y_values = [test_regression.resp_func_val(x_list=plan_point[:3], coeff_list=new_nat_coeffs) for plan_point in
                nat_matrix]
disp_ad = (m / (N - d)) * sum([dispersion(y_average_values[i], new_y_values_correct) for i in range(k)])
f_p = float(disp_ad / disp_b)
adequacy = f_t > f_p

print("Normalized orthogonal central composition plan:\n{}".format(norm_matrix_table))
print("{}Naturalized orthogonal central composition plan:\n{}".format("\n" * 2, nat_matrix_table))
print("Naturalized coefficients:" + (len(b_list) * " {:.4},").format(*b_list))
print("Y values:" + (len(new_y_values) * " {:.4},").format(*new_y_values))
print("New Y values:" + (len(new_y_values_correct) * " {:.4},").format(*new_y_values_correct))
print("New coefficients:" + (len(new_nat_coeffs) * " {:.4},").format(*new_nat_coeffs))
print("Adequacy of the regression equation: {}".format(adequacy))
