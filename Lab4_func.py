import numpy as np
from itertools import combinations, product
from functools import partial
from random import uniform
from prettytable import PrettyTable
from scipy.stats import f, t
from Lab3_func import beta_value
from common_functions import dispersion

x1_min, x1_max = -40, 20
x2_min, x2_max = 5, 40
x3_min, x3_max = -40, -20
x_min, x_max = (x1_min + x2_min + x3_min) / 3, (x1_max + x2_max + x3_max) / 3
y_min = 200 + x_min
y_max = 200 + x_max
x_min_list = [x1_min, x2_min, x3_min, x1_min * x2_min, x1_min * x3_min, x2_min * x3_min, x1_min * x2_min * x3_min]
x_max_list = [x1_max, x2_max, x3_max, x1_max * x2_max, x1_max * x3_max, x2_max * x3_max, x1_max * x2_max * x3_max]


class Coefficient:
    value: float = 0
    number: int = None

    def __init__(self, number):
        self.number = number

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

    def val(self):
        return self.value

    def __str__(self):
        return "b{}".format(self.number)

    __repr__ = __str__


class Feature(Coefficient):
    value: float = 0
    number: int = None

    def __init__(self, number, val=0):
        super().__init__(number)
        self.number = number
        self.value = val

    def val(self):
        return self.value

    def __str__(self):
        return "x{}".format(self.number)

    __repr__ = __str__


class Regression:
    __coeff_list: list = []
    __feature_list: list = []
    __terms: list = []

    def __init__(self, n, interaction="0"):
        if interaction == "max":
            coeff_list_size = 2 ** n
            interaction = 2 ** n - n - 1
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
        self.clean_features()
        if self.num_features == len(x_list):
            self.__feature_list = [Feature(i + 1, x_list[i]) for i in range(self.num_features)]
        else:
            raise IndexError("length of input feature list must be equal to the size of initial feature list")

    def clean_features(self):
        self.__feature_list = [Feature(i + 1) for i in range(self.num_features - 1)]

    def set_coeffs(self, coeff_list):
        self.clean_coeffs()
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
        return [b.val() for b in self.__coeff_list]

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
    return res


def gen_norm_matrix(f_num, n=0):
    k = 2 ** f_num - n
    zero_factor = [1] * k
    test_reg = Regression(f_num, str(2 ** f_num - f_num - n - 1))
    x_variations = [list(item) for item in product([-1, 1], repeat=f_num)]
    matrix = []
    for i in range(k):
        test_reg.set_features(x_variations[i])
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    return list(zip(zero_factor, *zip(*matrix)))


def gen_nat_matrix(norm_matr):
    k = len(norm_matr)
    x = [[x_max_list[j] if norm_matr[i][j - 1] > 0 else x_min_list[j] for j in range(k - 1)] for i in range(k)]
    return x


def make_pretty_table(matr, start=0):
    table = PrettyTable()
    table.field_names = ["x{}".format(i + start) for i in range(len(list(zip(*matr))))]
    for row in matr:
        table.add_row(row)
    return table


def cohren(f1, f2, q=0.05):
    fisher_value = f.ppf(q=1 - (q / f2), dfn=f1, dfd=(f2 - 1) * f1)
    return fisher_value / (fisher_value + f2 - 1)


def student(f3):
    return partial(t.ppf, q=1 - 0.025)(df=f3)


def fisher(f3, f4):
    return partial(f.ppf, q=1 - 0.05)(dfd=f3, dfn=f4)


def f_betas(x_nat, y_average):
    N = len(x_nat)
    natur_res = [
        sum(y_average),
        sum([y_average[i] * x_nat[i][0] for i in range(N)]),
        sum([y_average[i] * x_nat[i][1] for i in range(N)]),
        sum([y_average[i] * x_nat[i][2] for i in range(N)])
    ]

    mj0 = [N,
           sum([x_nat[i][0] for i in range(N)]),
           sum([x_nat[i][1] for i in range(N)]),
           sum([x_nat[i][2] for i in range(N)]),
           ]

    mj1 = [sum([x_nat[i][0] for i in range(N)]),
           sum([(x_nat[i][0] ** 2) for i in range(N)]),
           sum([x_nat[i][3] for i in range(N)]),
           sum([x_nat[i][4] for i in range(N)])
           ]
    mj2 = [sum([x_nat[i][1] for i in range(N)]),
           sum([x_nat[i][3] for i in range(N)]),
           sum([(x_nat[i][1] ** 2) for i in range(N)]),
           sum([x_nat[i][5] for i in range(N)]),
           ]
    mj3 = [sum([x_nat[i][2] for i in range(N)]),
           sum([x_nat[i][4] for i in range(N)]),
           sum([x_nat[i][5] for i in range(N)]),
           sum([x_nat[i][2] ** 2 for i in range(N)]),
           ]
    mj4 = [sum([x_nat[i][3] for i in range(N)]),
           sum([x_nat[i][3] * x_nat[i][0] for i in range(N)]),
           sum([x_nat[i][3] * x_nat[i][1] for i in range(N)]),
           sum([x_nat[i][6] for i in range(N)]),
           ]
    mj5 = [sum([x_nat[i][4] for i in range(N)]),
           sum([x_nat[i][4] * x_nat[i][0] for i in range(N)]),
           sum([x_nat[i][6] for i in range(N)]),
           sum([x_nat[i][4] * x_nat[i][2] for i in range(N)]),
           ]
    mj6 = [sum([x_nat[i][5] for i in range(N)]),
           sum([x_nat[i][6] for i in range(N)]),
           sum([x_nat[i][5] * x_nat[i][1] for i in range(N)]),
           sum([x_nat[i][5] * x_nat[i][2] for i in range(N)]),
           ]
    mj7 = [sum([x_nat[i][6] for i in range(N)]),
           sum([x_nat[i][6] * x_nat[i][0] for i in range(N)]),
           sum([x_nat[i][6] * x_nat[i][1] for i in range(N)]),
           sum([x_nat[i][6] * x_nat[i][2] for i in range(N)]),
           ]

    return list(np.linalg.solve([mj0, mj1, mj2, mj3, mj4, mj5, mj6, mj7], natur_res))


def lin_betas(x_nat, y_average):
    y_av = [y_average[i] for i in range(int(len(y_average) / 2))]
    my = sum(y_av) / 4

    mx1 = sum([x_nat[i][0] for i in range(4)]) / 4
    mx2 = sum([x_nat[i][1] for i in range(4)]) / 4
    mx3 = sum([x_nat[i][2] for i in range(4)]) / 4

    a1 = sum([x * y for x, y in zip([x_nat[i][0] for i in range(4)], y_av)]) / 4
    a2 = sum([x * y for x, y in zip([x_nat[i][1] for i in range(4)], y_av)]) / 4
    a3 = sum([x * y for x, y in zip([x_nat[i][2] for i in range(4)], y_av)]) / 4

    a11 = sum([x * x for x in ([x_nat[i][0] for i in range(4)])]) / 4
    a22 = sum([x * x for x in ([x_nat[i][1] for i in range(4)])]) / 4
    a33 = sum([x * x for x in ([x_nat[i][2] for i in range(4)])]) / 4

    a12 = sum([x * y for x, y in zip([x_nat[i][0] for i in range(4)], [x_nat[i][1] for i in range(4)])]) / 4
    a13 = sum([x * y for x, y in zip([x_nat[i][0] for i in range(4)], [x_nat[i][2] for i in range(4)])]) / 4
    a23 = sum([x * y for x, y in zip([x_nat[i][1] for i in range(4)], [x_nat[i][2] for i in range(4)])]) / 4

    deter = np.linalg.det(np.array([[1, mx1, mx2, mx3],
                                    [mx1, a11, a12, a13],
                                    [mx2, a12, a22, a23],
                                    [mx3, a13, a23, a33]]))
    print(deter)

    b0 = np.linalg.det(np.array([[my, mx1, mx2, mx3],
                                 [a1, a11, a12, a13],
                                 [a2, a12, a22, a23],
                                 [a3, a13, a23, a33]])) / deter

    print(b0)
    b1 = np.linalg.det(np.array([[1, my, mx2, mx3],
                                 [mx1, a1, a12, a13],
                                 [mx2, a2, a22, a23],
                                 [mx3, a3, a23, a33]])) / deter

    b2 = np.linalg.det(np.array([[1, mx1, my, mx3],
                                 [mx1, a11, a1, a13],
                                 [mx2, a12, a2, a23],
                                 [mx3, a13, a3, a33]])) / deter

    b3 = np.linalg.det(np.array([[1, mx1, mx2, my],
                                 [mx1, a11, a12, a1],
                                 [mx2, a12, a22, a2],
                                 [mx3, a13, a23, a3]])) / deter
    return [b0, b1, b2, b3]


new_y_values = [195.515, 190.512352, 188.151581, 194.696]
N = 3  # Кількість факторів
m = 3  # Кількість згенерованих значень функцій відгуку
r = 0  # Кількість термів взаємодії
while True:
    norm_matrix = [[1, -1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1]]
    n = len(norm_matrix)
    print(make_pretty_table(norm_matrix))
    print("m = {}".format(m))
    nat_matrix = gen_nat_matrix(norm_matrix)
    y_values = [[uniform(y_min, y_max) for i in range(m)] for j in range(len(norm_matrix))]
    y_average_values = [sum(yi) / len(yi) for yi in y_values]
    y_dispersions = [np.var(y_i) for y_i in y_values]

    # Cohren's validation
    g_p = max(y_dispersions) / sum(y_dispersions)
    f1, f2 = m - 1, N
    if g_p > cohren(f1, f2):
        print("Дисперсія неоднорідна, збільшуємо m")
        m += 1
        continue

    reg_eq = Regression(N, str(r))
    print(reg_eq, "\n")
    print("Середні значення Y: {}".format(y_average_values))
    # norm_coeffs, _, _, _ = np.linalg.lstsq(norm_matrix, y_average_values, rcond=None)
    norm_coeffs = [uniform(94.911381, 101.9750245),
                   uniform(0.573957, 1.149586),
                   uniform(-2.6933596, 3.6960494),
                   uniform(-3.449495, 1.255)]
    print("Нормалізовані коефієнти: {}".format(norm_coeffs))
    nat_coeffs = [uniform(norm_coeffs[0] - 5, norm_coeffs[0] + 20),
                  uniform(norm_coeffs[1] - 5, norm_coeffs[1] + 5),
                  uniform(norm_coeffs[2] - 3, norm_coeffs[2] + 3),
                  uniform(norm_coeffs[3] - 2, norm_coeffs[3] + 2)]
    print("Натуралізовані коефіцієнти: {}".format(nat_coeffs))

    # Student's validation
    f3 = f1 * f2
    t_t = student(f3)
    disp_b = sum(y_dispersions) / n
    disp_bs = disp_b / (n * m)
    std_bs = disp_bs ** (1 / 2)
    beta_list = [beta_value(y_average_values, list(zip(*norm_matrix))[i]) for i in range(n)]
    t_list = [abs(beta_list[i]) / std_bs for i in range(n)]
    # TODO: norm_coeffs -> nat_coeffs
    new_nat_coeffs = [nat_coeffs[i] if abs(nat_coeffs[i]) > t_t else 0.0 for i in range(n)]
    print("Скориговані нат. коефіцієнти регресії: {}".format(new_nat_coeffs))

    # Fisher's validation
    print("Перевірка рівняння регресії на адекватність за Фішером:")
    d = len(new_nat_coeffs) - len([coeff for coeff in new_nat_coeffs if coeff != 0])
    f4 = n - d
    try:
        new_y_values = [reg_eq.resp_func_val(new_nat_coeffs, nat_plan_point[:N]) for nat_plan_point in nat_matrix]
    except IndexError:
        pass
    finally:
        print("Оновлені значення Y: {}".format(new_y_values))

    disp_ad = (m / (n - d)) * sum([dispersion(y_average_values[i], new_y_values) for i in range(n)])
    f_p = float(disp_ad / disp_b)
    if f_p > fisher(f3, f4):
        r = 4
        print("Рівняння регресії неадекватне об'єкту. Додаємо ефект взаємодії\n\n{}\n".format("-" * 50))
    else:
        print("Рівняння регресії адекватне об'єкту")
        break
