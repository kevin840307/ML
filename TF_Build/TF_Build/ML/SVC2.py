import cvxopt
import cvxopt.solvers
import numpy as np
import matplotlib.pyplot as plt
from tool import margin_2D_point, make_meshgrid

class SVC():
    def fit(self, x, y):
        '''
        SVM formula:
        optimal(b, w)
        min 1/2 * sum_n(sum_m(a_n * a_m * y_n * y_m * z_n^T * z_m)) - sum_n(a_n)
        subject to sum_n(y_n(a_n)) = 0
                   (a_n) >= 0

        Quadratic Programming:
        optimal a←QP(Q, p, A, c)←QP(Q, p, A, c, B, b)
        min 1/2 * a^T * Q * a + p^T * a
        subject to a_m^T * u >= c_m


        objective function:
        Q = y_n * y_m * z_n^T * z_m
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
        
        Q =  cvxopt.matrix(np.dot(y, y.T) * np.dot(x, np.transpose(x)))
        p = cvxopt.matrix(-np.ones(x_len))
        A = cvxopt.matrix(-np.eye(x_len))
        c = cvxopt.matrix(np.zeros(x_len))
        B = cvxopt.matrix(np.reshape(y, (1, -1)))
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options['show_progress'] = False
        result = cvxopt.solvers.qp(Q, p, A, c, B, b)

        self.__alphas = np.array(result['x']).flatten()
        self.__sv = self.__alphas > 1e-6
        self.__w = np.sum(np.array(result['x'] * y * x), axis=0).reshape(-1)
        self.__support_vectors = x[self.__sv,:]
        self.__b = (1./ y[self.__sv][0]) - np.dot(self.__w, x[self.__sv, :][0])
        #self.__b = ((1./ y[self.__sv]) 
        #          - np.sum(
        #              self.__w.reshape(-1,1).T * x[self.__sv, :],
        #              axis=-1,
        #              keepdims=True)
        #          )[0]
        '''
        Vertical dist:
            |(ax + by + c)| / sqrt(a^2 + b^2)
        '''
        self.__margin = 1. /  np.linalg.norm(self.__w)
        

    def predict(self, x):
        pred = np.dot(np.reshape(self.__w, (1, -1)), x.T) + self.__b
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
    X = np.concatenate([np.random.randn(20, 2) - [8, 8], np.random.randn(20,2) + [2, 2]])
    Y = np.concatenate([np.ones(20) * -1, np.ones(20)])

    model = SVC()
    model.fit(X, Y)
    model.info()
    print(model.predict(X))

    w = model.get_w()
    b = model.get_b()

    # hyperplane
    x = np.linspace(-20, 20)
    y = -w[0] / w[1] * x - b / w[1]
    plt.plot(x, y)
    
    # support vectors hyperplane
    support_vectors = model.get_support_vectors()
    for support_vector in support_vectors:
        x = np.linspace(-20, 20)
        y = (-w[0] / w[1]) * x  + (support_vector[1] - (-w[0] / w[1]) * support_vector[0])
        plt.plot(x, y, 'k--')

        x, y = margin_2D_point(w, b, support_vector)
        plt.plot([support_vector[0], x], [support_vector[1], y], 'k--', c='r')

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='g')

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.gcf().set_size_inches(8, 8)
    plt.show()