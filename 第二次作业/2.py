import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    return np.exp(x[0] + 3 * x[1] - 0.1) + \
           np.exp(x[0] - 3 * x[1] - 0.1) + \
           np.exp(-x[0] - 0.1)


def df(x):
    return np.array([np.exp(x[0] + 3 * x[1] - 0.1) +
                     np.exp(x[0] - 3 * x[1] - 0.1) -
                     np.exp(-x[0] - 0.1),
                     3 * np.exp(x[0] + 3 * x[1] - 0.1) -
                     3 * np.exp(x[0] - 3 * x[1] - 0.1)])

def d2f(x):
    return np.array([[np.exp(x[0]+3*x[1]-0.1)+np.exp(x[0]-3*x[1]-0.1)+np.exp(-x[0]-0.1),
                      3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1)],
                     [3*np.exp(x[0]+3*x[1]-0.1)-3*np.exp(x[0]-3*x[1]-0.1),
                      9*np.exp(x[0]+3*x[1]-0.1)+9*np.exp(x[0]-3*x[1]-0.1)]])


def minf(x, alpha, beta, epsilon):
    while True:
        d = -df(x)
        if np.linalg.norm(df(x), ord=2) <= epsilon:
            break
        t = 1
        while f(x + t * d) > f(x) + alpha * t * np.transpose(df(x)).dot(d):
            t = beta * t
        x = x + t * d
    return f(x)

def cal(x, alpha, beta, num, epsilon=1e-5):
    errors = []
    while True:
        errors.append(f(x) - num)
        d = -df(x)
        if np.linalg.norm(df(x), ord=2) <= epsilon:
            break
        t = 1
        while f(x + t * d) > f(x) + alpha * t * np.transpose(df(x)).dot(d):
            t = beta * t
        x = x + t * d
    return errors


def main():
    x0 = np.array([0, 0])
    lens = []

    fmin = minf(x0.copy(), 0.5, 0.8, 1e-7)

    errors = cal(x0.copy(), 0.5, 0.8, fmin)
    lens.append(len(errors))
    plt.plot(errors, label='alpha=0.5, beta=0.8', marker = 'o', linestyle='-')

    errors = cal(x0.copy(), 0.5, 0.6, fmin)
    lens.append(len(errors))
    plt.plot(errors, label='alpha=0.5, beta=0.6', marker = 'o', linestyle='-')

    errors = cal(x0.copy(), 0.3, 0.8, fmin)
    lens.append(len(errors))
    plt.plot(errors, label='alpha=0.3, beta=0.8', marker = 'o', linestyle='-')

    errors = cal(x0.copy(), 0.3, 0.3, fmin)
    lens.append(len(errors))
    plt.plot(errors, label='alpha=0.2, beta=0.2', marker = 'o', linestyle='-')

    plt.xlabel('iteration / times')
    plt.ylabel('error')
    plt.title('The situation where the error changes with the number of iterations')
    plt.yscale("log")
    plt.legend()
    plt.xticks(range(0, max(lens) + 1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()
