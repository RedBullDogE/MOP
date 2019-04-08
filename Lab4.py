import numpy as np
from itertools import combinations

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

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
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
    coeff_list: list = []
    coeff_list_size: int = 0
    feature_list_size: int = 0
    feature_list: list = []
    terms: list = []

    def __init__(self, n, interaction="0"):
        if interaction == "max":
            self.coeff_list_size = 2 ** n
        else:
            self.coeff_list_size = n + int(interaction) + 1

        self.feature_list_size = n
        self.feature_list = [Feature(i + 1) for i in range(n)]
        self.coeff_list = [Coefficient(i) for i in range(self.coeff_list_size)]
        for i in range(self.feature_list_size):
            self.terms.extend(list(combinations(self.feature_list, i + 1)))

    def value_of_y(self, coeff_list, x_list):
        if self.coeff_list_size == len(coeff_list) and self.feature_list_size == len(x_list):
            self.set_features(x_list)
            self.set_coeffs(coeff_list)
            term_value_list = [sum(term) for term in self.terms]
            return sum(map(lambda b, x: b * x, self.coeff_list, term_value_list))
        else:
            raise IndexError("input lists must be the same size as the regression data lists")

    def set_features(self, x_list):
        for i in range(self.feature_list_size):
            self.feature_list[i].set_value(x_list[i])

    def clean_features(self):
        self.feature_list = [Feature(i + 1) for i in range(self.feature_list_size)]

    def set_coeffs(self, coeff_list):
        for i in range(self.coeff_list_size):
            self.coeff_list[i].set_value(coeff_list[i])

    def clean_coeffs(self):
        self.feature_list = [Coefficient(i) for i in range(self.coeff_list_size)]

    @property
    def terms_val(self):
        return [tuple(x.val() for x in term) for term in self.terms]

    @property
    def coeffs_val(self):
        return [b.val() for b in self.coeff_list]

    def __str__(self):
        cur_coeff = 0
        string_reg = "{}".format(self.coeff_list[cur_coeff])
        for i in range(self.feature_list_size):
            x_list = list(combinations(self.feature_list, i + 1))
            row = " + {}" + "*{}" * len(x_list[0])
            for j in range(len(x_list)):
                cur_coeff += 1
                string_reg += row.format(self.coeff_list[cur_coeff], *x_list[j])
        return string_reg


# def regression(b_list, x_list):
#     b_num = [0, 1, 2, 3, 12, 13, 23, 123]
#     b = {i: j for i, j in (b_num, b_list)}
#     x = {i: j for i, j in enumerate(x_list, start=1)}
#     y = b[0] + b[1] * x[1] + b[2] * x[2] + b[3] * x[3] + \
#         b[12] * x[1] * x[2] + b[13] * x[1] * x[3] + b[23] * x[2] * x[3] + \
#         b[123] * x[1] * x[2] * x[3]
#     return y


reg = Regression(2, "max")
print(reg)
reg.set_features([1, 2])
reg.set_coeffs([1, 2, 3, 4])
print(reg)


