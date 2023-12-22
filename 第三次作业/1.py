import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    res=0
    for i in x:
        res+=i*math.log(i)
    return res

def df(x):
    res=[]
    for i in x:
        res.append(math.log(i)+1)
    return np.array(res)

def d2f(x):
    res=[]
    for i in x:
        res.append(1/i)
    return np.diagflat(res)


def g(v, A, b):
    return (np.transpose(b) @ v + np.sum(np.exp(-np.transpose(A) @ v - 1))).item()

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

def main():

    n=100
    p=30
    A=np.random.rand(p,n)
    while np.linalg.matrix_rank(A)!=p:
        A=np.random.rand(p,n)*10
    x=np.random.rand(n)*0.1
    v=np.random.rand(p)*0.1
    b=np.matmul(A,x.reshape(n,1))
    alpha=0.2
    beta=0.8

    fmin = minf(x,alpha,beta,1e-7)

    errors1 = []

    #标准Newton方法
    x1=x.copy()
    print("可行初始点Newton方法")
    while True:
        errors1.append(f(x1))
        diff=df(x1)
        diff2=d2f(x1)
        d=np.matmul(np.linalg.inv(np.block([[diff2,A.transpose()],[A,np.zeros((p,p))]])),np.block([[-diff.reshape(n,1)],[np.zeros((p,1))]]))[:100]
        lambda2=np.matmul(np.matmul(d.reshape(1,n),diff2),d)
        if 0.5*lambda2<=0.0001:
            print("函数值:{}".format(f(x1)))
            print("x:{}".format(x1))
            break
        t=1
        while f(x1+t*d.reshape(n))>f(x1)-alpha*t*lambda2:
            t=beta*t
        x1+=t*d.reshape(n)

    errors1 -= fmin

    plt.plot(errors1, label='feasible: alpha=0.1, beta=0.7')

    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.title('feasible newton method')
    plt.yscale("log")
    plt.legend()
    plt.show()

    #不可行初始点Newton方法
    x2=x.copy()
    v2=v.copy()
    print("不可行初始点Newton方法")
    errors2 = []
    while True:
        errors2.append(f(x2))
        diff=df(x2)
        diff2=d2f(x2)
        r=np.block([[diff.reshape(n,1)+np.matmul(A.transpose(),v2.reshape(p,1))],[np.matmul(A,x2.reshape(n,1))-b]])
        d=np.matmul(np.linalg.inv(np.block([[diff2,A.transpose()],[A,np.zeros((p,p))]])),-r)
        dx=d[:100]
        dv=d[100:]
        if np.linalg.norm(r)<=0.0001:
            print("函数值:{}".format(f(x2)))
            print("x:{}".format(x2))
            print("拉格朗日乘子v:{}".format(v2))
            break
        while np.linalg.norm(np.block([[df(x2+t*dx.reshape(n)).reshape(n,1)+np.matmul(A.transpose(),(v2.reshape(p,1)+t*dv))],
                                       [np.matmul(A,(x2.reshape(n,1))+t*dx)-b]]))>(1-alpha*t)*np.linalg.norm(r):
            t=beta*t
        x2+=t*dx.reshape(n)
        v2+=t*dv.reshape(p)

    errors2 -= fmin
    plt.plot(errors2, label='infeasible: alpha=0.1, beta=0.7')
    plt.yscale("log")
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.legend()
    plt.title('infeasible newton method')
    plt.show()

    #对偶Newton方法
    x3=x.copy()
    v3=v.copy()
    print("对偶Newton方法")
    value = []
    while True:
        value.append(-g(v3, A,b))
        tmp=-np.matmul(A.transpose(),v3.reshape(p,1))-1
        grad_=np.array([math.exp(i) for i in tmp])
        grad=b-np.matmul(A,grad_.reshape(n,1))
        hessian=np.matmul(np.matmul(A,np.diagflat(grad_)),A.transpose())
        d=-np.matmul(np.linalg.inv(hessian),grad)
        lambda2=np.matmul(np.matmul(grad.transpose(),np.linalg.inv(hessian)),grad)
        if 0.5*lambda2<=0.0001:
            print("函数值:{}".format(-np.matmul(b.transpose(), v3.reshape(p, 1)).item() - grad_.sum()))
            print("x:{}".format(x3))
            print("拉格朗日乘子v:{}".format(v3))
            break
        t=1
        while np.matmul(b.transpose(),v3.reshape(p,1)+t*d)+\
                sum([math.exp(i) for i in -np.matmul(A.transpose(),v3.reshape(p,1)+t*d)-1])>=\
                np.matmul(b.transpose(),v3.reshape(p,1))+sum([math.exp(i) for i in -np.matmul(A.transpose(),
                                                                                              v3.reshape(p,1))-1])+alpha*t*np.matmul(grad.reshape(1,p),d):
            t=beta*t
        v3+=t*d.reshape(p)

    value = fmin - value
    plt.plot(value, label='dual: alpha=0.1, beta=0.7')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.title('dual newton method')
    plt.show()


if __name__ == '__main__':
    main()