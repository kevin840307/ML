import cvxopt
import cvxopt.solvers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles  
from sklearn.datasets import make_moons 
from tool import margin_2D_point, gaussian_kernel, make_meshgrid

class SVC():
    def __init__(self, gamma=1., C=1):
        self.gamma = gamma
        self.C = C

    def fit(self, x, y):
        '''
        SVM formula:
        K = kernel
        K(x, x') = z_n^T * z_m
        optimal(b, w)
        min 1/2 * sum_n(sum_m(a_n * a_m * y_n * y_m * K(x, x'))) - sum_n(a_n)
        subject to sum_n(y_n(a_n)) = 0
                   0 <= (a_n) <= C
        
        sv = (a_s > 0)
        free_sv = (a_s > 0) & (a_s < C)

        Quadratic Programming:
        optimal a←QP(Q, p, A, c)←QP(Q, p, A, c, B, b)
        min 1/2 * a^T * Q * a + p^T * a
        subject to a_m^T * u >= c_m


        objective function:
        K(x, x') = z_n^T * z_m
        Q = y_n * y_m * K(x, x')
        p = -1_N

        A and c are N conditions
        B and b are a condition
        constraints:
        A = n-th unit direction
        c = 0
        B = 1_n
        b = 0
        M = N = data size


        Correspondence cvxopt op:
        P = Q
        q = p
        G = A
        h = c
        B = A
        a = u
        '''
        assert ((len(x) == len(y))), "size error"
        assert ((len(x) == len(y)) & (len(x) > 0)), "input x error"

        x_len = len(x)
        dimension = len(x[0])
        y = np.reshape(y, (-1, 1))
        kernel_x = gaussian_kernel(x, x, self.gamma)

        Q =  cvxopt.matrix(np.dot(y, y.T) * kernel_x)
        p = cvxopt.matrix(-np.ones(x_len))
        A = cvxopt.matrix(np.concatenate([-np.eye(x_len), np.eye(x_len)]))
        c = cvxopt.matrix(np.concatenate([np.zeros(x_len), np.ones(x_len) * self.C]))
        B = cvxopt.matrix(np.reshape(y, (1, -1)))
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options['show_progress'] = False
        result = cvxopt.solvers.qp(Q, p, A, c, B, b)

        self.__alphas = np.array(result['x']).flatten()
        self.__sv = self.__alphas > 1e-6
        self.__free_sv = (self.__alphas > 1e-6) & (self.__alphas < self.C)
        self.__w = np.sum(np.array(result['x'] * y * x), axis=0).reshape(-1)
        self.__support_vectors = x[self.__sv,:]
        self.__free_support_vectors = x[self.__free_sv,:]
        self.__a_y = np.reshape(self.__alphas[self.__sv], (-1, 1)) * np.reshape(y[self.__sv], (-1, 1))
        self.__free_a_y = np.reshape(self.__alphas[self.__free_sv], (-1, 1)) * np.reshape(y[self.__free_sv], (-1, 1))
        self.__b =  np.sum(y[self.__free_sv]) 
        self.__b -= np.sum(self.__free_a_y * gaussian_kernel(self.__free_support_vectors, x[self.__free_sv], self.gamma))
        self.__b /= len(self.__free_support_vectors)

        '''
        Vertical dist:
            |(ax + by + c)| / sqrt(a^2 + b^2)
        '''
        self.__margin = 1. /  np.linalg.norm(self.__w)



    def predict(self, x):
        pred = np.sum(self.__a_y * gaussian_kernel(self.__support_vectors, x, self.gamma), axis=0) + self.__b
        pred_sign = np.sign(pred)
        return pred_sign

    def get_w(self):
        return self.__w

    def get_b(self):
        return self.__b

    def get_support_vectors(self):
        return self.__support_vectors

    def get_margin(self):
        return self.__margin

    def info(self):
        print('margin:', self.__margin)
        print('support vectors:', self.__support_vectors)
        print('support vectors len:', len(self.__support_vectors))

if __name__ == '__main__':
    np.random.seed(1)
    X, Y = make_circles(n_samples=100,factor=0.5,noise=0.1)
    Y[Y == 0] = -1
    X = np.array(X).astype(float) * 10
    Y = np.array(Y).astype(float)
    #np.random.shuffle(Y)

    model = SVC(1, 2147483647)
    model.fit(X, Y)
    model.info()
    print(model.predict(X))

    w = model.get_w()
    b = model.get_b()


    X0, X1 = np.linspace(-20, 20), np.linspace(-20, 20)
    xx, yy = make_meshgrid(X0, X1)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # support vectors hyperplane
    support_vectors = model.get_support_vectors()
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    #plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='g')

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.gcf().set_size_inches(8, 8)
    plt.show()