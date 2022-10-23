"""practice using gradient descent to perform a polynomial regression of some simple functions"""

import numpy as np
import matplotlib.pyplot as plt

def base_func(input):
    return((input ** 2) + input - 3) #quadratic

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

def eval_polynomial(ws, b, x):
    output = 0
    for i in range(np.size(ws)[0]):
        output += ws[i] * (x ** (i + 1))
    output += b
    return(output)

def cost(ws, b, x_targ, y_targ):
    length = np.size(x_targ)[0]
    J = 0

    for i in range(length):
        J += (eval_polynomial(ws, b, x_targ[i]) - y_targ[i]) ** 2
    J /= 2 * length
    return(J)


def mean_unscale(scld_val, range, avg):
    """takes a mean scaled value and returns the unscaled value"""
    return(scld_val * range + avg)

def gradient_descent(ws, bias, alpha, xn_normed, y_targ):
    print(xn_normed, '\n')
    print(xn_normed[:, 0])

def poly_reg(x_targ, y_targ, degree):
    x1_norm, x1_n_rng, x1_n_avg = mean_normed(x_targ, 1)
    x2_norm, x2_n_rng, x2_n_avg = mean_normed(x_targ, 2)

    xn_normed = np.empty((degree, np.shape(x_targ)[0]))
    xn_rng = np.empty(degree)
    xn_avg = np.empty(degree)
    for i in range(degree):
        temp_data, temp_rng, temp_avg = mean_normed(x_targ, i + 1)
        xn_normed[i] = temp_data
        xn_rng[i] = temp_rng
        xn_avg = temp_avg


    # print("manual __________________________ \n", x1_norm[:4])
    # print(x2_norm[:4])
    # print("loop created ____________________ \n", xn_normed[0][:4])
    # print(xn_normed[1][:4])
    weights = np.zeros(degree)
    bias = 0
    alpha = .001


    num_iters = 1000
    cost_hist = np.empty(num_iters)

    gradient_descent(weights, bias, alpha, xn_normed, y_targ)



def main():
    x_targ = np.arange(-10, 10)
    y_targ = create_target(x_targ)
    # x1_norm, x1_n_rng, x1_n_avg = mean_normed(x_targ, 1)
    # x2_norm, x2_n_rng, x2_n_avg = mean_normed(x_targ, 2)

    degree = 2
    poly_reg(x_targ, y_targ, degree)

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
