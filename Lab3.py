from random import uniform
from Lab3_func import calc_norm_coeff, response_function
from Lab2_func import output_table


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

m = 3

# y_values = [[uniform(y_min, y_max) for i in range(m)] for j in range(len(norm_plan_matrix))]
y_values = [
    [15, 18, 16],
    [10, 19, 13],
    [11, 14, 12],
    [16, 19, 16]
]
y_values = list(map(lambda x: list(map(lambda y: float(y), x)), y_values))
y_average_values = [sum(yi) / len(yi) for yi in y_values]

norm_coeff = calc_norm_coeff(nat_plan_matrix, y_average_values)
y_result_norm = [response_function(norm_coeff, nat_plan_point) for nat_plan_point in nat_plan_matrix]

print(output_table(nat_plan_matrix, y_values, y_average_values))

print(y_result_norm)


