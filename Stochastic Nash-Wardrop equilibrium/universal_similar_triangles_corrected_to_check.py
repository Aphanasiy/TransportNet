from math import sqrt
import numpy as np
from numba import jit, njit
import torch

@jit
def universal_similar_triangles_function(phi_big_oracle, phi_small_solver, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    iter_step = 5
    L_value = L_init
    A_previous = 0.0
    A_current = 0.0
    u_parameter = np.copy(t_start)
    t_parameter = np.copy(t_start)
    y_parameter = np.copy(t_parameter)
    
    u_previous = u_parameter
    t_previous = t_parameter
    
    flows_weighted = np.zeros(len(t_start))
    flows_weighted_previous = flows_weighted
    entropy_weighted = 0.0
    entropy_weighted_previous = entropy_weighted
    
    duality_gap_init = 0.0
    epsilon_absolute = 0.0
    #epsilon_absolute = epsilon

    
    primal_func_history = []
    inner_iters_history = []

    if verbose:
        duality_gap_history = []
    
    for counter in range(max_iter):
        inner_iters_num = 1
        while True:
            alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_previous / L_value)
            A_current = A_previous + alpha

            y_parameter = (alpha * u_previous + A_previous * t_previous) / A_current
            phi_grad_y_parameter = phi_big_oracle.grad(y_parameter)
            phi_small_solver.update(alpha, phi_grad_y_parameter)

            u_parameter = phi_small_solver.argmin_function(u_start = u_previous)
            t_parameter = (alpha * u_parameter + A_previous * t_previous) / A_current

            flows_weighted = (A_previous * flows_weighted_previous - 
                              alpha * phi_grad_y_parameter) / A_current
            entropy_weighted = (A_previous * entropy_weighted + 
                              alpha * phi_big_oracle.entropy(y_parameter)) / A_current   

            if counter == 0 and inner_iters_num == 1:
                primal_func_value = primal_dual_oracle.sigma_sum_func(flows_weighted) - \
                                    entropy_weighted
                dual_func_value = primal_dual_oracle.dual_func_value(t_parameter)
                duality_gap_init = primal_func_value + dual_func_value
                
                print('primal_init = ' + str(primal_func_value))
                print('dual_init = ' + str(dual_func_value))
                print('duality_gap_init = ' + str(duality_gap_init))
                epsilon_absolute = epsilon * duality_gap_init

            left_value = (phi_big_oracle.func(y_parameter) + 
                          np.dot(phi_grad_y_parameter, t_parameter - y_parameter) + 
                          0.5 * alpha / A_current * epsilon_absolute) - phi_big_oracle.func(t_parameter)
            right_value = - 0.5 * L_value * np.sum(np.square(t_parameter - y_parameter))
            if (left_value >= right_value):
                break
            else:
                L_value *= 2
                phi_small_solver.undo_update(alpha, phi_grad_y_parameter)
                print('iteration_num = ' + str(counter + 1) + ': L_value = ' + str(L_value))
                inner_iters_num += 1

                    
        A_previous = A_current
        L_value /= 2
        
        t_previous = t_parameter
        u_previous = u_parameter
        flows_weighted_previous = flows_weighted
        entropy_weighted_previous = entropy_weighted
        
        primal_func_value = primal_dual_oracle.sigma_sum_func(flows_weighted) - \
                            entropy_weighted
        primal_func_history.append(primal_func_value)
        duality_gap = primal_func_value + \
                      primal_dual_oracle.dual_func_value(t_parameter)
        
        duality_gap_history.append(duality_gap)
        inner_iters_history.append(inner_iters_num)
        
        if duality_gap < epsilon_absolute:
            result = {'times': t_parameter,
                      'flows': flows_weighted,
                      'iter_num': counter + 1,
                      'duality_gap_history': duality_gap_history,
                      'res_msg': 'success',
                      'inner_iters_history': inner_iters_history,
                      'primal_func_history': primal_func_history}
            if verbose:
                print('Success! Iterations number: ' + str(counter + 1))
                print('Primal_func_value = ' + str(primal_func_value))
                print('Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
                print('Phi big oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time))
                
            return result  

        
        if verbose and ((counter + 1) % iter_step == 0 or counter == 0):
            print('Iterations number: ' + str(counter + 1))
            print('Dual_gap_value = ' + str(duality_gap))
            print('Primal_func_value = ' + str(primal_func_value))
            print('Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
            #print('Duality_gap = ' + str(duality_gap))
            
    result = {'times': t_parameter,
              'flows': flows_weighted,
              'iter_num': counter + 1,
              'duality_gap_history': duality_gap_history,
              'res_msg': 'iterations number exceeded',
              'inner_iters_history': inner_iters_history,
              'primal_func_history': primal_func_history}
    
    if verbose:
        print('Primal_func_value = ' + str(primal_func_value))
        print('Iterations number exceeded!')
    return result


#@njit
def universal_similar_triangles_function_with_torch(
                                         phi_big_oracle, phi_small_solver, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    
    iter_step = 5
    L_value = torch.FloatTensor([L_init])
    A_previous = torch.FloatTensor([0.0])
    A_current = torch.FloatTensor([0.0])
    u_parameter = torch.from_numpy(t_start)
    t_parameter = torch.from_numpy(t_start)
    y_parameter = t_parameter.clone()
    
    u_previous = u_parameter.clone()
    t_previous = t_parameter.clone()
    
    # Making tensor and inline filling it
    flows_weighted = torch.FloatTensor(len(t_start)).zero_()
    flows_weighted_previous = flows_weighted.clone()
    entropy_weighted = torch.FloatTensor([0.0])
    entropy_weighted_previous = entropy_weighted.clone()
    
    duality_gap_init = torch.FloatTensor([0.0])
    epsilon_absolute = torch.FloatTensor([0.0])
    #epsilon_absolute = epsilon

    
    primal_func_history = []
    inner_iters_history = []

    if verbose:
        duality_gap_history = []
    
    for counter in range(max_iter):
        inner_iters_num = 1
        while True:
            alpha = torch.add(
                        torch.div(torch.FloatTensor([0.5]), L_value),
                        torch.sqrt( torch.add(
                            torch.div(
                                torch.FloatTensor([0.25]),
                                torch.square(L_value)),
                            torch.div(
                                A_previous, 
                                L_value))))
            A_current = torch.add(A_previous, alpha)
            y_parameter = torch.div(
                            torch.add(
                              torch.mul(alpha, u_previous), 
                              torch.mul(A_previous, t_previous) ), 
                            A_current)
            phi_grad_y_parameter = torch.from_numpy(phi_big_oracle.grad(y_parameter.numpy())) # To numpy converting
            phi_small_solver.update(alpha.numpy(), phi_grad_y_parameter.numpy()) # To numpy converting
            u_parameter = torch.from_numpy(phi_small_solver.argmin_function(u_start = u_previous.numpy())) # To numpy   # Correct returning num?
            t_parameter = torch.div(
                            torch.add(
                              torch.mul(alpha, u_parameter),
                              torch.mul(A_previous, t_previous) ),
                            A_current)
            flows_weighted = torch.div(
                                torch.sub(
                                  torch.mul(A_previous, flows_weighted_previous), 
                                  torch.mul(alpha, phi_grad_y_parameter)),
                                A_current)
            entropy_weighted = torch.div(
                                  torch.add(
                                    torch.mul(A_previous, entropy_weighted),
                                    torch.mul(alpha, phi_big_oracle.entropy(y_parameter.numpy()))), #To numpy
                                  A_current)
            if counter == 0 and inner_iters_num == 1:
                primal_func_value = torch.sub(
                    primal_dual_oracle.sigma_sum_func(flows_weighted),
                    entropy_weighted) # From numpy
                dual_func_value = primal_dual_oracle.dual_func_value(t_parameter.numpy()) # To numpy
                duality_gap_init = torch.add(primal_func_value, dual_func_value)
                
                print(' > primal_init = ' + str(float(primal_func_value)))
                print(' > dual_init = ' + str(float(dual_func_value)))
                print(' > duality_gap_init = ' + str(float(duality_gap_init)))
                epsilon_absolute = torch.mul(epsilon, duality_gap_init)
            left_value = torch.sub(
                          torch.add(torch.add(
                            phi_big_oracle.func(y_parameter.numpy()), #To/From numpy
                            torch.dot(phi_grad_y_parameter, torch.sub(t_parameter, y_parameter))),
                            torch.mul(torch.mul(
                                torch.FloatTensor([0.5]),
                                torch.div(
                                    alpha,
                                    A_current)),
                                epsilon_absolute)),
                          phi_big_oracle.func(t_parameter.numpy())) #numpy

            right_value = torch.mul(torch.mul(
                                torch.FloatTensor([-0.5]),
                                L_value),
                                torch.sum(torch.square(torch.sub(t_parameter, y_parameter))))
            if (left_value >= right_value):
                break
            
            L_value.mul_(2) #inline
            phi_small_solver.undo_update(alpha.numpy(), phi_grad_y_parameter.numpy()) # To numpy
            print('iteration_num = ' + str(counter + 1) + ': L_value = ' + str(float(L_value)))
            inner_iters_num += 1
                    
        A_previous = A_current.clone()
        L_value.div_(2)
        
        t_previous = t_parameter.clone()
        u_previous = u_parameter.clone()
        flows_weighted_previous = flows_weighted.clone()
        entropy_weighted_previous = entropy_weighted.clone()
        
        primal_func_value = torch.sub(
                                primal_dual_oracle.sigma_sum_func(flows_weighted.numpy()), #numpy
                                entropy_weighted)
        primal_func_history.append(primal_func_value)
        duality_gap = torch.add(
                        primal_func_value,
                        primal_dual_oracle.dual_func_value(t_parameter.numpy())) #numpy
        
        duality_gap_history.append(duality_gap)
        inner_iters_history.append(inner_iters_num)
        
        if duality_gap < epsilon_absolute:
            result = {'times': t_parameter,
                      'flows': flows_weighted,
                      'iter_num': counter + 1,
                      'duality_gap_history': duality_gap_history,
                      'res_msg': 'success',
                      'inner_iters_history': inner_iters_history,
                      'primal_func_history': primal_func_history}
            if verbose:
                print(' | Success! Iterations number: ' + str(counter + 1))
                print(' | Primal_func_value = ' + str(primal_func_value))
                print(' | Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
                print(' | Phi big oracle elapsed time: {:.0f} sec'.format(phi_big_oracle.time)) 
                
            return result  

        
        if verbose and ((counter + 1) % iter_step == 0 or counter == 0):
            print(' - Iterations number: ' + str(counter + 1))
            print(' - Dual_gap_value = ' + str(duality_gap))
            print(' - Primal_func_value = ' + str(primal_func_value))
            print(' - Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
            #print('Duality_gap = ' + str(duality_gap))
            
    result = {'times': t_parameter,
              'flows': flows_weighted,
              'iter_num': counter + 1,
              'duality_gap_history': duality_gap_history,
              'res_msg': 'iterations number exceeded',
              'inner_iters_history': inner_iters_history,
              'primal_func_history': primal_func_history}
    
    if verbose:
        print(' - Primal_func_value = ' + str(primal_func_value))
        print(' | Iterations number exceeded!')
    return result