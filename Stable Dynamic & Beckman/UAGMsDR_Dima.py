import numpy as np
from numpy.linalg import norm
from math import sqrt


def ternarySearch(f, l, r, eps):
    """
    If r == inf, minumum shouldn't be at point l.
    :param f: a function
    :param l: left point, must be a non-negative number 
    :param r: right point, must be a non-negative number, could be inf
    :param eps: precision
    :return: argmin of f on the interval [l,r]
    """
    if r == float('Inf'):
        r = 100
#         r = 1
#         # while f(l) > f(r):
#         #    r *= 2
#         while f(r) > f(2*r):
#             # l = r
#             r = 2*r
#             print(l, r, ', f ', f(l), f(r))
#         r = 2*r
    while r - l > eps:
        print(l, r, '| f ', f(l), f(r))
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        if f(m1) < f(m2):
            r = m2
        else:
            l = m1
    return l


# def V(d, x, z):
#     """
#     Bregman divergance
#     :param d: prox-function
#     :param x: point x
#     :param z: point z
#     :return: Bregman divergance
#     """
#     return d(x) - d(z) - np.dot(d.grad(z), x - z) # need to define grad and scalar product

def quadratic_equation_solution(phi_big_oracle, x, y, A, eps):
    """
    Solution of ax^2+bx+c=0
    :return: float, the largest x
    """
    # all variables are updated so we can drop indices
    # need to specify * norm
    f = phi_big_oracle.func
    g = phi_big_oracle.grad

    a = norm(g(y))**2
    b = 2*f(x)-2*f(y)-eps
    c = 2*A*(f(x)-f(y))
    D = b**2 - 4*a*c
    # return the bigger root
    return (-b + sqrt(D)) / (2 * a)


def UAGMsDR(phi_big_oracle, prox_h, primal_dual_oracle,
            t_start, L_init=None, max_iter=1000,
            crit_name='dual_gap_rel', beta=0.0001, h=0.0001, eps=1e-5, eps_abs=None, verbose=False):
    # we don't need L_init but but leave it to keep the same code structure
    """
    :param eps: accuracy
    :return: x^k
    """

    if crit_name == 'dual_gap_rel':
        def crit():
            nonlocal duality_gap, duality_gap_init, eps
            return duality_gap < eps * duality_gap_init
    if crit_name == 'dual_rel':
        def crit():
            nonlocal dual_func_history, eps
            l = len(dual_func_history)
            return dual_func_history[l // 2] - dual_func_history[-1] \
                < eps * (dual_func_history[0] - dual_func_history[-1])
    if crit_name == 'primal_rel':
        def crit():
            nonlocal primal_func_history, eps
            l = len(primal_func_history)
            return primal_func_history[l // 2] - primal_func_history[-1] \
                < eps * (primal_func_history[0] - primal_func_history[-1])

    duality_gap_init = None

    primal_func_history = []
    dual_func_history = []
    # inner_iters_history = []
    duality_gap_history = []

    success = False
    iter_step = 10

    ### 0 ###
    f = phi_big_oracle.func
    g = phi_big_oracle.grad

#     print(phi_big_oracle.func)
#     for i in range(20):
#         print(i, phi_big_oracle.func(i*t_start))

    grad_sum = np.zeros(len(t_start))
    # y_start = v_prev = np.copy(x_start)
    ### 0 ###

    ### 1 ###
    k = 0
    A = 0
    # v0 в статье не определено, правильно ли я понимаю, что это должно быть
    # argmin_{x \in E} \psi0(x) = argmin_{x \in E} V(x,x0)
    x = y = v = np.copy(t_start)  # need to define v
    # psi = lambda t: (1/2)*norm(t-x)**2 # V(x, x0)  # it is in prox_h
    ### 1 ###

    ### 2, 8, 9 ###
    for k in range(1, max_iter + 1):
        # or start from 0?
        ### 2, 8, 9 ###

        ### 3 ###
        # beta = ternarySearch(lambda b: f(v + b * (x - v)), 0, 1, eps)
        # beta = 0.0001
        y = v + beta * (x - v)
        ### 3 ###

        ### 4 ###
        f_grad_y = g(y)
        # f_y_hashtag = f_grad_y/norm(f_grad_y) #####
        f_y_hashtag = f_grad_y
        # h = ternarySearch(lambda h: f(y - h * f_y_hashtag),
        #                   0, float('Inf'), eps) #####
        # h = 0.0001
        x = y - h * f_y_hashtag
        a = quadratic_equation_solution(
            phi_big_oracle, x, y, A, eps)  # quadratic equation
        ### 4 ###

        ### 5 ###
        A = A + a
        ### 5 ###

        ### 6 ###
        # psi = lambda t: psi(t) + a * (f(y) + np.dot(f_grad_y, t - y))
        # we don't use because it is in grad_sum
        ### 6 ###

        ### 7 ###
        # same as with argmin in universal_triangles because of similar structure
        grad_sum = grad_sum + a * f_grad_y
        v = prox_h(y - grad_sum, A, u_start=v)
        ### 7 ###

        ### OUTPUT ###
        if k == 1:
            flows_weighted = - grad_sum / A
            duality_gap_init = primal_dual_oracle.duality_gap(
                x, flows_weighted)
#             if eps_abs is None:
#                 eps_abs = eps * duality_gap_init

            if verbose:
                print('Primal_init = {:g}'.format(
                    primal_dual_oracle.primal_func_value(flows_weighted)))
                print('Dual_init = {:g}'.format(
                    primal_dual_oracle.dual_func_value(v)))
                print('Duality_gap_init = {:g}'.format(duality_gap_init))

        # A_prev = A
        # L_value /= 2

        # t_prev = t
        # u_prev = u
        # grad_sum_prev = grad_sum
        flows_weighted = - grad_sum / A

        primal_func_value = primal_dual_oracle.primal_func_value(
            flows_weighted)
        dual_func_value = primal_dual_oracle.dual_func_value(v) # x
        duality_gap = primal_dual_oracle.duality_gap(v, flows_weighted) # v

        primal_func_history.append(primal_func_value)
        dual_func_history.append(dual_func_value)
        duality_gap_history.append(duality_gap)
        # inner_iters_history.append(inner_iters_num)

        # if duality_gap < eps_abs:
        #    success = True
        #    break

        if verbose and (k == 1 or k % iter_step == 0):
            print('\nIterations number: ' + str(it_counter))
            # print('Inner iterations number: ' + str(inner_iters_num))
            print('Primal_func_value = {:g}'.format(primal_func_value))
            print('Dual_func_value = {:g}'.format(dual_func_value))
            print('Duality_gap = {:g}'.format(duality_gap))
            print('Duality_gap / Duality_gap_init = {:g}'.format(
                duality_gap / duality_gap_init), flush=True)

        result = {'times': v,
                  'flows': flows_weighted,
                  'iter_num': k,
                  'duality_gap_history': duality_gap_history,
                  # 'inner_iters_history': inner_iters_history,
                  'primal_func_history': primal_func_history,
                  'dual_func_history': dual_func_history,
                  }

        if success:
            result['res_msg'] = 'success'
        else:
            result['res_msg'] = 'iterations number exceeded'

        if verbose:
            if success:
                print('\nSuccess! Iterations number: ' + str(k))
            else:
                print('\nIterations number exceeded!')
        print('Primal_func_value = {:g}'.format(primal_func_value))
        print(
            'Duality_gap / Duality_gap_init = {:g}'.format(duality_gap / duality_gap_init))
        print('Phi_big_oracle elapsed time: {:.0f} sec'.format(
            phi_big_oracle.time))
        # print('Inner iterations total number: ' + str(sum(inner_iters_history)))

    return result
