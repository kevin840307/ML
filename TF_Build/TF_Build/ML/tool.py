import numpy as np

def margin_2D_point(w, b, p):
    '''
    Let:
        ax + by + c = 0 # hyperplane formula
        m1 * m2 = -1
        m1 = -a / b
        m2 = b / a
        p1 = (p_x, p_y) = (support_vectors[0, 0], support_vectors[0, 1])

    Vertical Formula:
        Formula:
            y - p_y = m2 * (x - p_x)

        Point of transform:
            x = (y - p_y) / m2 + p_x
            y = m2 * (x - p_x) + p_y

    Intersection Point Formula:
        Formula:
            y_1 = f(x)
            y_2 = g(x)
            y_1 = y_2
            f(x) = g(x)

            x = (-c - b * y) / a
            y = (-c - a * x) / b
            f(x) = (-c1 - a1 * x) / b1
            g(x) = (-c2 - a2 * x) / b2

            x = (((-c2 - a2 * x) / b2) * b1 + c1) / - a1
            y = (((-c2 - b2 * y) / a2) * a1 + c1) / - b1

        Point of transform:
            x = (((-c2 - a2 * x) / b2) * b1 + c1) / - a1
            y = (((-c2 - b2 * y) / a2) * a1 + c1) / - b1

        Ex:
            hyperplane y : y = m1 * x - b / w[1]
              vertical y : y = m2 * (x - p_x) + p_y

            x  = (-b / w[1] + m2 * p_x - p_y) / (m2 - m1)
            y = m2 * (x - p_x) + p_y

    '''
    assert ((len(w) == 2) & (len(p) == 2)), "size error"

    m1 = -w[0] / w[1]
    m2 = w[1] / w[0]

    x = (-b / w[1] + m2 * p[0] - p[1]) / (m2 - m1)
    y = m2 * (x - p[0]) + p[1]

    return x, y


def polynomial_kernel(x0, x1, zeta=1., gamma=1., Q=2):
    assert zeta >= 0, "zeta error"
    assert gamma > 0, "gamma error"

    return (zeta + gamma * np.dot(x0, np.transpose(x1))) ** Q


def gaussian_kernel(x0, x1, gamma=1.):

    assert gamma > 0, "gamma error"

    return np.exp(-gamma * np.linalg.norm(np.expand_dims(x0, axis=1) - x1, axis=-1) ** 2)

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy
