"""practice using gradient descent to perform a polynomial regression of some simple functions"""

import numpy as np
import matplotlib.pyplot as plt

def base_func(input):
    # return((input ** 2) + input - 3) #quadratic
    return(3 * input ** 3 - 10 * input ** 2 + input + 20)

def create_target(x_targ):
    """x_targ is numpy array"""
    # print(x_targ.shape())
    y_targ = np.empty_like(x_targ)
    for i in range(len(y_targ)):
        y_targ[i] = base_func(x_targ[i])

    return(y_targ)

def mean_normed(x_targ, power):
    """returns a mean normalized n degree array of the original array for polynomial regression"""
    size = x_targ.shape[0]
    # print("size = ", size)
    xn = x_targ ** power
    min = np.min(xn)
    max = np.max(xn)
    den = max - min
    avg = np.average(xn)
    # print("testting ggngngn", xn)

    xn_norm = np.empty_like(xn, dtype=np.float64)
    for i in range(size):
        xn_norm[i] = (xn[i] - avg) / den
        # print((xn[i] - avg) / den)
        # print(xn_norm[i], "--")

    # print(xn_norm, "\n _____________")
    # normalized array, range, avg (so we can recover real vals from scaled weights)
    return(xn_norm, den, avg)

# def eval_polynomial(ws, b, xn_normed):
#     output = 0
#     # print(type(ws))
#     # print(np.size(ws))
#     for i in range(np.size(ws)):
#         output += ws[i] *
#     output += b
#     return(output)

def cost(ws, bias, xn_normed, y_targ):
    length = np.size(y_targ)
    J = 0

    for i in range(length):
        func = bias
        for j in range(np.shape(ws)[0]):
            func += ws[j] * xn_normed[j][i]
        J += (func - y_targ[i]) ** 2
    J /= 2 * length
    return(J)


def mean_unscale(scld_val, range, avg):
    """takes a mean scaled value and returns the unscaled value"""
    return(scld_val * range + avg)

def gradient_descent(ws, bias, alpha, reg_term, xn_normed, y_targ):
    # print(xn_normed, '\n')
    # print(xn_normed[:, 0])
    new_ws = np.empty_like(ws)
    # for each partial derivative WRT weight_i
    for i in range(np.shape(ws)[0]):
        dJ_dwi = 0
        # to sum up over all training data
        for j in range(np.shape(y_targ)[0]):
            func = bias
            # evaluating function by each weight (in increasing degree)
            for k in range(np.shape(ws)[0]):
                func += ws[k] * xn_normed[k][j]
            dJ_dwi += (func - y_targ[j]) * xn_normed[i][j]
        dJ_dwi /= np.shape(y_targ)[0]
        regulariztion = (reg_term * ws[i]) / np.shape(y_targ)[0]
        new_ws[i] = ws[i] - (alpha * (dJ_dwi + regulariztion))

    dJ_db = 0
    for y in range(np.shape(y_targ)[0]):
        func = bias
        for h in range(np.shape(ws)[0]):
            func += ws[h] * xn_normed[h][y]
        dJ_db += func - y_targ[y]
    dJ_db /= np.shape(y_targ)[0]

    new_bias = bias - (alpha * dJ_db)

    return(new_ws, new_bias)


def eval_polynomial(xn_normed, weights, bias):
    """xn should be an array of the scaled x value at increasing powers of x
    using the scaling of each power of x"""
    total = bias
    for i in range(np.shape(weights)[0]):
        total += xn_normed[i] * weights[i]

    return(total)

def create_scaled_xs(x_vals, degree, xn_rng, xn_avg):
    """creates an array of the corresponding scaled values of ascending powers of x"""
    output = np.zeros((np.shape(x_vals)[0], degree))
    for i in range(np.shape(x_vals)[0]):
        for j in range(degree):
            temp = x_vals[i] ** (j + 1)
            temp -= xn_avg[j]
            temp /= xn_rng[j]
            output[i][j] = temp

    return(output)

def poly_reg(x_targ, y_targ, degree, alpha, reg_term, num_iters):
    # x1_norm, x1_n_rng, x1_n_avg = mean_normed(x_targ, 1)
    # x2_norm, x2_n_rng, x2_n_avg = mean_normed(x_targ, 2)

    # print(np.shape(x_targ))
    xn_normed = np.empty((degree, np.shape(x_targ)[0]))
    # creates array for [x, x^2, x^3...]
    xn_rng = np.empty(degree)
    xn_avg = np.empty(degree)
    for i in range(degree):
        temp_data, temp_rng, temp_avg = mean_normed(x_targ, i + 1)
        xn_normed[i] = temp_data
        xn_rng[i] = temp_rng
        xn_avg[i] = temp_avg

    weights = np.zeros(degree)
    # weights = [2, 0, 1]
    bias = 0

    cost_hist = np.empty(num_iters)
    # weight_hist = np.zeros((num_iters, degree + 1))
    # order in arr will be [b, w0, w1...]
    # weight_hist[0][0] = np.float64(bias)
    # we are initializing weights at the origin

    for i in range(num_iters):
        cost_hist[i] = cost(weights, bias, xn_normed, y_targ)
        # weight_hist[i] =
        weights, bias = gradient_descent(weights, bias, alpha, reg_term, xn_normed, y_targ)

    print(cost_hist[num_iters - 1])
    print("degree =", degree, "alpha =", alpha, " lambda = ", reg_term)
    # print(cost_hist[9000], "cost at iter=9000")
    print(bias, weights)

    xs_to_plot = np.linspace(-10, 10, num=100, dtype=np.float64)
    scld_xs = create_scaled_xs(xs_to_plot, degree, xn_rng, xn_avg)
    # print(np.shape(scld_xs))
    predicted_vals = np.zeros(np.shape(xs_to_plot))
    # print(np.shape(predicted_vals))
    # print(np.shape(xs_to_plot)[0])
    for j in range(np.shape(xs_to_plot)[0]):
        # print(j)
        predicted_vals[j] = eval_polynomial(scld_xs[j], weights, bias)

    # print(weights, bias)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.arange(num_iters), cost_hist)
    ax1.set_title("Cost vs iters")
    ax2.scatter(x_targ, y_targ)
    ax2.plot(xs_to_plot, predicted_vals, 'r')
    plt.show()


def main():
    x_targ = np.arange(-10, 10)
    y_targ = create_target(x_targ)
    # x1_norm, x1_n_rng, x1_n_avg = mean_normed(x_targ, 1)
    # x2_norm, x2_n_rng, x2_n_avg = mean_normed(x_targ, 2)

    degree = 3
    alpha = .1
    reg_term = .01
    num_iters = 10000
    poly_reg(x_targ, y_targ, degree, alpha, reg_term, num_iters)

    # print(x_targ)
    # print(x1_norm)
    # print(x2_norm)
    # print(y_targ)

    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.scatter(x_targ, y_targ)
    # ax2.scatter(x1_norm, y_targ)
    # plt.show()




if __name__ == '__main__':
    main()
