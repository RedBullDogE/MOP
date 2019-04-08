import numpy as np
from itertools import combinations, product
from random import uniform
from prettytable import PrettyTable

x1_min, x1_max = -40, 20
x2_min, x2_max = 5, 40
x3_min, x3_max = -40, -20
x_min, x_max = (x1_min + x2_min + x3_min) / 3, (x1_max + x2_max + x3_max) / 3
y_min = 200 + x_min
y_max = 200 + x_max


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

    def val(self):
        return self.value

    def __str__(self):
        return "x{}".format(self.number)

    __repr__ = __str__


class Regression:
    __coeff_list: list = []
    __coeff_list_size: int = 0
    __feature_list_size: int = 0
    __feature_list: list = []
    __terms: list = []

    def __init__(self, n, interaction="0"):
        if interaction == "max":
            self.__coeff_list_size = 2 ** n
            interaction = 2 ** n - n - 1
        else:
            try:
                interaction = int(interaction)
                if interaction > 2 ** n - n - 1:
                    raise ValueError("interaction must be less than 2 ** n - n")
                self.__coeff_list_size = n + interaction + 1
            except TypeError:
                raise TypeError("interaction must be str(int) or 'max'")

        self.__feature_list_size = n
        self.__feature_list = [Feature(i + 1) for i in range(n)]
        self.__coeff_list = [Coefficient(i) for i in range(self.__coeff_list_size)]
        for i in range(min(self.__feature_list_size, interaction + 1)):
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
        if self.__feature_list_size == len(x_list):
            for i in range(self.__feature_list_size):
                self.__feature_list[i].set_value(x_list[i])
        else:
            raise IndexError("length of input feature list must be equal to the size of initial feature list")

    def clean_features(self):
        self.__feature_list = [Feature(i + 1) for i in range(self.__feature_list_size)]

    def set_coeffs(self, coeff_list):
        if self.__coeff_list_size == len(coeff_list):
            for i in range(self.__coeff_list_size):
                self.__coeff_list[i].set_value(coeff_list[i])
        else:
            raise IndexError("length of input coefficient list must be equal to the size of initial coefficient list")

    def clean_coeffs(self):
        self.__feature_list = [Coefficient(i) for i in range(self.__coeff_list_size)]

    @property
    def terms_list(self):
        return [tuple(x.val() for x in term) for term in self.__terms]

    @property
    def coeffs_list(self):
        return [b.val() for b in self.__coeff_list]

    @property
    def num_features(self):
        return self.__feature_list_size

    @property
    def num_coeffs(self):
        return self.__coeff_list_size

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


def gen_norm_matrix(f_num):
    k = 2 ** f_num
    zero_factor = [1] * k
    test_reg = Regression(f_num, "max")
    x_variations = [list(item) for item in product([-1, 1], repeat=f_num)]
    matrix = []
    for i in range(k):
        test_reg.set_features(x_variations[i])
        row = list(map(mul, test_reg.terms_list))
        matrix.append(row)

    return list(zip(zero_factor, *zip(*matrix)))


N = 3

# reg = Regression(N, "max")
# print(reg)
# reg.set_features([1, 2, 3])
# print(reg.terms_list)
mat = gen_norm_matrix(N)


def makePrettyTable(matr):
    table = PrettyTable()
    table.field_names = ["x{}".format(i) for i in range(len(matr))]
    for row in matr:
        table.add_row(row)
    return table


print(makePrettyTable(mat))
