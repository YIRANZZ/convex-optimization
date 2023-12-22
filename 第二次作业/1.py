import numpy as np
import math
import matplotlib.pyplot as plt

def cal(r, epsilon):
    k = 0
    points = []
    while True:
        xk = ((r - 1) / (r + 1)) ** k * np.array([r, (-1) ** k])
        points.append(xk)
        if np.linalg.norm(xk, ord=2) <= epsilon:
            break
        k += 1

    x1 = [point[0] for point in points]
    x2 = [point[1] for point in points]

    plt.figure(figsize=(8, 6))

    plt.plot(x1, x2, marker = 'o', linestyle='-', color='g', label='Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-r-0.5, r+0.5)
    plt.ylim(-5, 5)

    plt.title(f"r={r}, iteration : {len(points)-1} times")

    plt.show()

def main():

    epsilon = 1e-5

    cal(1, epsilon)

    cal(0.5, epsilon)
    cal(2, epsilon)

    cal(10, epsilon)
    cal(0.01, epsilon)


if __name__ == '__main__':
    main()
