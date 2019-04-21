from itertools import combinations, product
from prettytable import PrettyTable


x1_min, x1_max = -2, 4
x2_min, x2_max = -10, 8
x3_min, x3_max = -3, 6
x_min, x_max = (x1_min + x2_min + x3_min) / 3, (x1_max + x2_max + x3_max) / 3
y_min = 200 + x_min
y_max = 200 + x_max
x_min_list = [x1_min, x2_min, x3_min, x1_min * x2_min, x1_min * x3_min, x2_min * x3_min, x1_min * x2_min * x3_min]
x_max_list = [x1_max, x2_max, x3_max, x1_max * x2_max, x1_max * x3_max, x2_max * x3_max, x1_max * x2_max * x3_max]


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
    return res


def gen_norm_matrix(f_num, n=0, quadratic=False):
    k = 2 ** f_num - n
    zero_factor = [1] * k
    test_reg = Regression(f_num, 2 ** f_num - f_num - n - 1, quadratic)
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


def make_pretty_table(matr):
    table = PrettyTable()
    header = ["x0", "x1", "x2", "x3", "x1 * x2", "x1 * x3", "x2 * x3", "x1 * x2 * x3", "x1 ^ 2", "x2 ^ 2", "x3 ^ 2"]
    table.field_names = header[:len(matr[0])]
    for row in matr:
        table.add_row(row)
    return table


norm_matrix = make_pretty_table(gen_norm_matrix(3, 0, True))
print(norm_matrix)
