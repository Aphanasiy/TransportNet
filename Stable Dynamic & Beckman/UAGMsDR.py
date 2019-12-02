from numpy.linalg import norm
from math import sqrt

def ternarySearch(f, l, r, eps):
    """
    If r == inf, minumum shouldn't be at point 0.
    :param f: a function
    :param l: left point, must be a number
    :param r: right point, could be inf
    :param eps: precision
    :return: argmin of f on the interval [l,r]
    """
    if r == float('Inf'):
        r = 1
        while f(0) > f(r):
            r *= 2
    while r - l > eps:
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if f(m1) < f(m2):
            r = m2
        else:
            l = m1
    return l


def V(d, x, z):
    """
    Bregman divergance
    :param d: prox-function
    :param x: point x
    :param z: point z
    :return: Bregman divergance
    """
    return d(x) - d(z) - np.dot(d.grad(z), x - z) # need to define grad and scalar product

def quadratic_equation_solution(f, x, y, A, eps):
    """
    Solution of ax^2+bx+c=0
    :return: float, the largest x
    """
    # all variables are updated so we can drop indices
    # need to specify * norm
    a = norm(f.grad(y))**2
    b = 2*f(x)-2*f(y)-eps
    c = 2 * A * (f(x)-f(y))
    D = b**2 - 4 * a * c
    # return the bigger root
    return (-b + sqrt(D)) / (2 * a)

def UAGMsDR(phi_big_oracle, prox_h, x_start, eps=1e-5):
    """
    :param eps: accuracy
    :return: x^k
    """
    ### 0 ###
    f = phi_big_oracle
    grad_sum_prev = np.zeros(len(x_start))
    y_start = v_prev = np.copy(x_start)
    ### 0 ###

    ### 1 ###
    k = 0
    A = 0
    # v0 в статье не определено, правильно ли я понимаю, что это должно быть
    # argmin_{x \in E} \psi0(x) = argmin_{x \in E} V(x,x0)
    x = v  # need to define v
    psi = V(x, x0)  # need to define V, x0
    ### 1 ###

    ### 2, 8, 9 ###
    max_num = 100
    for k in range(max_num):
    ### 2, 8, 9 ###

        ### 3 ###
        beta = ternarySearch(lambda beta: f(v + b * (x - v)), 0, 1, eps)
        y = v + beta * (x - v)
        ### 3 ###

        ### 4 ###
        f_grad_y = f.grad(y)
        f_y_hashtag = f_grad_y/norm(f_grad_y)
        h = ternarySearch(lambda h: f(y - h * f_y_hashtag), 0, float('Inf'), eps)
        x = y - h * f_y_hashtag
        a = quadratic_equation_solution(f, x, y, A, eps) # quadratic equation
        ### 4 ###

        ### 5 ###
        A = A + a
        ### 5 ###

        ### 6 ###
        psi = lambda x: psi(x) + a * (f(y) + np.dot(f_grad_y, x - y))
        ### 6 ###

        ### 7 ###
        # same as with argmin in universal_triangles because of similar structure
        grad_sum = grad_sum_prev + a * f_grad_y
        v = prox_h(y_start - grad_sum, A, u_start = v_prev)
        ### 7 ###


# f = lambda x: (0.8-x)**2

# print(ternarySearch(f, 0, float('Inf'), 0.0001))
