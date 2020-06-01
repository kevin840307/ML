import cvxopt
import cvxopt.solvers
import numpy as np
import matplotlib.pyplot as plt
from tool import margin_2D_point

class SVC():
    def fit(self, x, y):
        '''
        SVM formula:
        optimal(b, w)
        min 1/2 * w^T * w
        subject to y_n(w^T * x_n + b) >= 1


        Quadratic Programming:
        optimal u←QP(Q, p, A, c)
        min 1/2 * u^T * Q * u + p^T * u
        subject to a_m^T * u >= c_m


        objective function:
        u = [b, w]^T
        Q = [[0, 0_d^T],
             [0_d, I_d]]
        p = 0_(d + 1)


        constraints:
        a_n^T = y_n[1, x_n^T]
        c_n = 1
        M = N


        Correspondence cvxopt op:
        P = Q
        q = p
        G = A
        h = c
        x = u
        '''
        assert ((len(x) == len(y))), "size error"
        assert ((len(x) == len(y)) & (len(x) > 0)), "input x error"

        x_len = len(x)
        b_x = np.concatenate([np.ones((x_len, 1)), x], axis=-1)
        dimension = len(b_x[0])

        eye_process = np.eye(dimension)
        eye_process[0][0] = 0
        Q = cvxopt.matrix(eye_process)
        p = cvxopt.matrix(np.zeros(dimension))
        A = cvxopt.matrix(np.reshape(y, (-1, 1)) * b_x * -1)
        c = cvxopt.matrix(np.ones(x_len) * -1)
        cvxopt.solvers.options['show_progress'] = False
        result = cvxopt.solvers.qp(Q, p, A, c)

        u = np.array(result['x'])
        self.__alphas = np.array(result['z']).flatten()
        self.__sv = self.__alphas > 1e-6
        self.__w = u[1:].flatten()
        self.__b = u[0]
        self.__support_vectors = x[self.__sv,:]

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
    X = np.concatenate([np.random.randn(20, 2) - [7, 5], np.random.randn(20, 2) + [2, 2]])
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
