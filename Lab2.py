from random import uniform, gauss
from math import sqrt
from MOP.Lab2_func import *

# Input values. Variant of the task:
x1_min = -15
x1_max = 30
x2_min = 25
x2_max = 65
y_min = 70
y_max = 170

plan_matrix = [[x1_min, x2_min],
               [x1_max, x2_min],
               [x1_min, x2_max]]
norm_plan_matrix = [[-1.0, -1.0],
                    [1.0, -1.0],
                    [-1.0, 1.0]]

N = len(norm_plan_matrix)  # Number of plan points
probability = 0.99  # required accuracy

# Table for finding Romanovsky criterion
m_list = {1: 2, 2: 6, 3: 8, 4: 10, 5: 12, 6: 15, 7: 20}
r_cr_list = {
    0.99: [1.73, 2.16, 2.43, 2.62, 2.75, 2.9, 3.08],
    0.98: [1.72, 2.13, 2.37, 2.54, 2.66, 2.8, 2.96],
    0.95: [1.71, 2.1, 2.27, 2.41, 2.52, 2.64, 2.78],
    0.9: [1.69, 2.0, 2.17, 2.29, 2.39, 2.49, 2.62]
}
global y_values, y_average_values, y_dispersions, major_deviation, f_list, theta_list, r_list

exit_flag = False
m_number = 2

while m_number < 8:
    y_values = [[uniform(y_min, y_max) for i in range(m_list[m_number])] for j in range(N)]
    y_average_values = [(sum(y_i) / len(y_i)) for y_i in y_values]

    y_dispersions = [dispersion(y_average_values[i], y_values[i]) for i in range(len(y_values))]
    major_deviation = 2 * sqrt((m_list[m_number] - 1) / (m_list[m_number] * (m_list[m_number] - 4)))
    f_list = f_coeffs(y_dispersions)
    theta_list = list(map(lambda x: (m_list[m_number] - 2) / m_list[m_number] * x, f_list))
    r_list = list(map(lambda x: abs(x - 1) / major_deviation, theta_list))
    for r in r_list:
        if r > r_cr_list[probability][m_number - 1]:
            m_number += 1
            exit_flag = False
            break
        else:
            exit_flag = True
            continue

    if exit_flag:
        break

if exit_flag:
    norm_coeffs = calc_norm_coeff(norm_plan_matrix, y_average_values)
    y_result_norm = [response_function(norm_coeffs, norm_plan_matrix[i]) for i in range(N)]

    # Naturalization
    nat_coeffs = calc_nat_coeff(x1_min, x1_max, x2_min, x2_max, norm_coeffs)
    y_result_nat = [response_function(nat_coeffs, plan_matrix[i]) for i in range(N)]

    print(output_table(norm_plan_matrix, y_values, y_average_values))
    print("\n{:15} {:8.5}, {:8.5}, {:8.5}".format("Dispersions:", *y_dispersions) +
          "\n{} {:8.5}".format("Major Deviation:", major_deviation) +
          "\n{:15} {:8.5}, {:8.5}, {:8.5}".format("F:", *f_list) +
          "\n{:15} {:8.5}, {:8.5}, {:8.5}".format("θ:", *theta_list) +
          "\n{:15} {:8.5}, {:8.5}, {:8.5}".format("R:", *r_list) +
          "\n{}".format("-" * 70) +
          "\n{:.5} + {:.5}*{} + {:.5}*{} = {}".format(norm_coeffs[0], norm_coeffs[1], "x1", norm_coeffs[2], "x2", "Y") +
          "\n{:35} {:8.5}, {:8.5}, {:8.5}".format("Y result (for normalized values):", *y_result_norm) +
          "\n{:.5} + {:.5}*{} + {:.5}*{} = {}".format(nat_coeffs[0], nat_coeffs[1], "x1", nat_coeffs[2], "x2", "Y") +
          "\n{:35} {:8.5}, {:8.5}, {:8.5}".format("Y result (for naturalized values):", *y_result_nat))
else:
    print("Can't find suitable values of M")
