from random import uniform
from Lab3_func import (calc_coeffs, response_function, dispersion, beta_value)
from output_functions import output_table, output_line

x1_min = -15.0
x1_max = 30.0
x2_min = 25.0
x2_max = 65.0
x3_min = -15.0
x3_max = -5.0

x_min = (x1_min + x2_min + x3_min) / 3
x_max = (x1_max + x2_max + x3_max) / 3

y_min = 200 + x_min
y_max = 200 + x_max

m = 3  # Number of experiments
N = 4  # Number of plan points
q = 0.05  # accuracy for Cochren's table

g_table = {
    0.05: {
        4: [0.9065, 0.7679, 0.6841, 0.6287, 0.5892, 0.5598]
    },
    0.01: {
        4: [0.9676, 0.8643, 0.7814, 0.7212, 0.6761, 0.6410]
    }
}

t_table = {
    4: 2.776,
    8: 2.306,
    12: 2.179,
    16: 2.120
}

f_table = {
    4: [7.7, 6.9, 6.6],
    8: [5.3, 4.5, 4.1],
    12: [4.8, 3.9, 3.5],
    16: [4.5, 3.6, 3.2]
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

# --------- Generating y values: y_min <= y <= y_max ---------
# Calculating naturalized coefficients for regression equation
y_values = [[uniform(y_min, y_max) for i in range(m)] for j in range(len(norm_plan_matrix))]
y_average_values = [sum(yi) / len(yi) for yi in y_values]

nat_coeffs = calc_coeffs(nat_plan_matrix, y_average_values)
y_nat_result = [response_function(nat_coeffs, nat_plan_point) for nat_plan_point in nat_plan_matrix]

# ------------ Cochren's criteria ------------
g_t = g_table[q][f2][f1]
y_dispersion_list = [dispersion(y_average_values[i], y_values[i]) for i in range(len(y_values))]
g_p = max(y_dispersion_list) / sum(y_dispersion_list)

# ------------ Student's criteria ------------
t_t = t_table[f3]

disp_b = sum(y_dispersion_list) / N
disp_bs = disp_b / (N * m)
std_bs = disp_bs ** (1 / 2)
beta_list = [beta_value(y_average_values, list(zip(*norm_plan_matrix))[i]) for i in range(N)]
t_list = [abs(beta_list[i]) / std_bs for i in range(N)]
new_nat_coeffs = [nat_coeffs[i] if nat_coeffs[i] > t_t else 0.0 for i in range(N)]

# ------------- Fisher's criteria -------------
d = len(new_nat_coeffs) - len([coeff for coeff in new_nat_coeffs if coeff != 0])
f4 = N - d
f_t = f_table[2 * N][f4 - 1]

new_y_values = [response_function(new_nat_coeffs, nat_plan_point) for nat_plan_point in nat_plan_matrix]
disp_ad = (m / (N - d)) * sum([dispersion(y_average_values[i], new_y_values) for i in range(N)])
f_p = float(disp_ad / disp_b)

print(output_table(nat_plan_matrix, y_values, y_average_values))
print(output_line("Y result (for naturalized values):", y_nat_result))
print("f1 = m - 1 = {}\nf2 = N = {}\nf3 = f1 * f2 = {}\nf4 = N - d = {}".format(f1, f2, f3, f4))
print(output_line("G-cochren's criteria (dispersion homogeneity)\n\tG_p < G_t:", g_p < g_t))
print("T-Student's criteria (significance of coefficients)")
print(output_line("\tY = {:.5} + {:.5} * x1 + {:.5} * x2 + {:.5} * x3\n\tNew Y values:".format(*new_nat_coeffs),
                  new_y_values))
print(output_line("F-Fisher's criteria (Fisher distribution)\n\tF_p < F_t:", f_p < f_t))
