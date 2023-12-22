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

def newton(x, alpha, beta, num, epsilon):
    errors = []
    while True:
        d = df(x)
        d2 = d2f(x)
        errors.append(f(x) - num)
        d = -np.linalg.inv(d2).dot(d)
        lambda2 = d.transpose().dot(d2).dot(d)
        if 0.5 * lambda2 <= epsilon:
            break
        t = 1
        while f(x + t * d) > f(x) - alpha * t * lambda2:
            t = beta * t
        x = x + t * d
    return errors

def main():

    x1 = np.array([1, 1], dtype = float)
    fmin = minf(x1.copy(), 0.1, 0.7, 1e-7)
    errors = newton(x1, 0.1, 0.7, fmin, 1e-5)
    plt.plot(errors, label='alpha=0.1, beta=0.7', marker='o', linestyle='-')
    plt.xlabel('iteration / times')
    plt.ylabel('error')
    plt.yscale("log")
    plt.title('The situation where the error changes with the number of iterations')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()
