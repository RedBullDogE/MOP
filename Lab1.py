import random


# Response function:
# Y = a0 + a1 * X1 + a2 * X2 + a3 * X3
def response_function(a_list: list, x_list: list):
    return a_list[0] + sum([a * x for a, x in zip(a_list[1:], x_list)])


# Normalizing X value to range from -1 (min) to 1 (max)
def normalize(x_min, x_max, x):
    x0 = (x_min + x_max) / 2
    dx = x_max - x0
    return (x - x0) / dx


# reg_coeff_str = input("Enter regression coefficients:")
# reg_coeff = list(map(reg_coeff_str.split(), float()))
regression_coefficients = [7, 2, 1, 3]

# Planning matrix (X1/2/3) and responsible function values (Y)
planning_matrix = [[random.uniform(0, 10) for i in range(3)] for j in range(8)]
y_values = [response_function(regression_coefficients, plan_point) for plan_point in planning_matrix]

factor_values = list(zip(*planning_matrix))  # Values of each factor separately
zero_values = [(min(factor) + max(factor)) / 2 for factor in factor_values]
y_standard = response_function(regression_coefficients, zero_values)
planning_matrix_normalized = list(zip(
    *[[normalize(min(factor_values[i]), max(factor_values[i]), planning_matrix[j][i]) for j in range(8)] for i in
      range(3)]))

# List of (Y - Ys) ^ 2 values
selection_criterion = [(y - y_standard) ** 2 for y in y_values]
index_criterion = selection_criterion.index(max(selection_criterion))  # index of max selection criterion value

# Additional Task:
# find and output the value of Y to the right of average Y
y_average_val = sum(y_values) / len(y_values)
y_values_sort = sorted(y_values)
y_difference = [yi - y_average_val for yi in y_values_sort if (yi - y_average_val) < 0]
print(y_values_sort[y_difference.index(max(y_difference))])

print("Regression equation:\n" +
      " Y = {} + {}*X1 + {}*X2 + {}*X3\n".format(regression_coefficients[0],
                                                 regression_coefficients[1],
                                                 regression_coefficients[2],
                                                 regression_coefficients[3]))
print("Planning matrices (ordinary/normalized) and values of response function:\n" +
      "{:^3} | {:^10} {:^10} {:^10} | {:^7.5} | {:^10} {:^10} {:^10}\n"
      .format("", "X1", "X2", "X3", "Y", "Xn1", "Xn2", "Xn3") +
      "\n".join(["{:^3} | {:>10.5} {:>10.5} {:>10.5} | {:>7.5} | {:>10.5} {:>10.5} {:>10.5}"
                .format(i + 1, planning_matrix[i][0], planning_matrix[i][1], planning_matrix[i][2],
                        y_values[i], planning_matrix_normalized[i][0],
                        planning_matrix_normalized[i][1], planning_matrix_normalized[i][2])
                 for i in range(8)]))
print("\nStandard value of Y:\n\tYs({:.5}, {:.5}, {:.5}) = {:.5}".format(*zero_values, y_standard))
print("Individual task:\n\tmax((Y - Ys)^2) = {:.5}".format(max(selection_criterion)))
print("Optimal parameters for a given criterion:\n\t{:.5} {:.5} {:.5}".format(*planning_matrix[index_criterion]))
print("Value of response function with the given factors:\n\t{:.5}".format(y_values[index_criterion]))
