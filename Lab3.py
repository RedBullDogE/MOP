from random import uniform
from Lab3_func import (calc_coeffs, response_function, dispersion, beta_value)
from output_functions import output_table, output_line

x1_min = -25.0
x1_max = 75.0
x2_min = 5.0
x2_max = 40.0
x3_min = 15.0
x3_max = 25.0

x_min = (x1_min + x2_min + x3_min) / 3
x_max = (x1_max + x2_max + x3_max) / 3

y_min = 10.0
y_max = 20.0
# y_min = 200 + x_min
# y_max = 200 + x_max

m = 3
N = 4
g_t = 0.7679
t_t = 2.306
# f_t = 4.1

f_table = {
    8: [5.3, 4.5, 4.1],
    12: [4.8, 3.9, 3.5]
}

f1 = m - 1
f2 = N
f3 = f1 * f2

norm_plan_matrix = [
    [1, -1, -1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, 1],
    [1, 1, 1, -1]
]

nat_plan_matrix = [
    [x1_min, x2_min, x3_min],
    [x1_min, x2_max, x3_max],
    [x1_max, x2_min, x3_max],
    [x1_max, x2_max, x3_min]
]

# y_values = [[uniform(y_min, y_max) for i in range(m)] for j in range(len(norm_plan_matrix))]
y_values = [
    [15, 18, 16],
    [10, 19, 13],
    [11, 14, 12],
    [16, 19, 16]
]
y_values = list(map(lambda x: list(map(lambda y: float(y), x)), y_values))
y_average_values = [sum(yi) / len(yi) for yi in y_values]

nat_coeffs = calc_coeffs(nat_plan_matrix, y_average_values)
y_nat_result = [response_function(nat_coeffs, nat_plan_point) for nat_plan_point in nat_plan_matrix]

# Cochren's criteria
y_dispersion_list = [dispersion(y_average_values[i], y_values[i]) for i in range(len(y_values))]
g_p = max(y_dispersion_list) / sum(y_dispersion_list)

# Student's criteria
disp_b = sum(y_dispersion_list) / N
disp_bs = disp_b / (N * m)
std_bs = disp_bs ** (1 / 2)
beta_list = [beta_value(y_average_values, list(zip(*norm_plan_matrix))[i]) for i in range(N)]
t_list = [abs(beta_list[i]) / std_bs for i in range(N)]
new_nat_coeffs = [nat_coeffs[i] if nat_coeffs[i] > t_t else 0 for i in range(N)]

# Fisher's criteria
d = len(new_nat_coeffs) - len([coeff for coeff in new_nat_coeffs if coeff != 0])
f4 = N - d

new_y_values = [response_function(new_nat_coeffs, nat_plan_point) for nat_plan_point in nat_plan_matrix]
disp_ad = (m / (N - d)) * sum([dispersion(y_average_values[i], new_y_values) for i in range(N)])
f_p = disp_ad / disp_b
f_t = f_table[2 * N][f4 - 1]
print(f_p < f_t)

print(output_table(nat_plan_matrix, y_values, y_average_values))
print(output_line("Y result (for naturalized values)", y_nat_result))
print("{}: {}".format("Dispersion homogeneity", g_p < g_t))
