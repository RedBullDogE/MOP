def output_table(x_matrix, y_matrix, y_average):
    """
    Converting matrix of X, matrix of Y and list of average values of Y into
    a string, that represents the table

    :param x_matrix: matrix of X
    :param y_matrix: matrix of Y
    :param y_average: list of average values of Y
    :return: string for visualizing information
    """
    title = "Planning matrix:\n"
    column_names = ["X{}".format(i + 1) for i in range(len(x_matrix[0]))] + \
                   ["Y{}".format(i + 1) for i in range(len(y_matrix[0]))] + \
                   ["Yav"]
    total_list = [x_matrix[i] + y_matrix[i] + [y_average[i]] for i in range(len(x_matrix))]

    x_body_layout = len(x_matrix[0]) * "{:^8.5} " + "|| "
    y_body_layout = len(y_matrix[0]) * "{:^8.5} " + "| "
    y_av_body_layout = "{:^8.5}\n"
    body_layout = x_body_layout + y_body_layout + y_av_body_layout
    body = body_layout.format(*column_names)
    for i in range(len(total_list)):
        body += body_layout.format(*total_list[i])

    res_str = title + body
    return res_str


def output_line(string, data):
    if isinstance(data, list):
        if isinstance(data[0], float):
            layout = "{}" + " {:5.5}," * len(data)
            result = layout[:-1].format(string, *data)
        else:
            layout = "{}" + " {:5}," * len(data)
            result = layout[:-1].format(string, *data)
    elif isinstance(data, (bool, int)):
        layout = "{} {}"
        result = layout.format(string, data)
    elif isinstance(data, float):
        layout = "{} {:5.5}"
        result = layout.format(string, data)
    else:
        raise TypeError()
    return result
