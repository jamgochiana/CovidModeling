'''
This file contains the functions used to estimate
the gamma parameters of our models
'''


import numpy as np
import cvxpy as cp
#import sklearn
import math


"""
U - healthy actively infected
V - hospital ish infections

"""



def estimate_params2(V, U):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
        
    T = len(V)

    gammas = np.ones(2)
    iters = 300
    for i in range(iters):
        grad = np.zeros(2)
        for t in range(T-1):
            grad[0] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[0] - V[t+1])
            grad[1] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[1] - U[t+1])
        gammas = gammas - grad * (0.5 / math.sqrt(i+1)) / np.linalg.norm(grad)


    return gammas


def estimate_params4(V, U):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
    T = len(V)

    gammas = np.ones((2, 2))
    iters = 1000
    for i in range(iters):
        grad = np.zeros((2, 2))
        for t in range(T-1):
            grad[0, 0] += V[t] * (V[t] * gammas[0, 0] - V[t+1] + gammas[0, 1] * U[t])
            grad[0, 1] += U[t] * (U[t] * gammas[0, 1] - V[t+1] + gammas[0, 0] * V[t])

            grad[1, 0] += V[t] * (V[t] * gammas[1, 0] - U[t+1] + gammas[1, 1] * U[t])
            grad[1, 1] += U[t] * (U[t] * gammas[1, 1] - U[t+1] + gammas[1, 0] * V[t])

        gammas = gammas - grad * (0.1 / math.sqrt(i+1)) / np.linalg.norm(grad)
    return gammas

'''
The functions that penalize the objective with L_1
'''



def l1_estimate_params2(V, U, lbd):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
    T = len(V)

    gammas = np.ones((T-1, 2))
    iters = 400
    for i in range(iters):
        grad = np.zeros((T, 2))
        for t in range(T-1):
            grad[t, 0] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[t, 0] - V[t+1])
            grad[t, 1] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[t, 1] - U[t+1])

            if t == T-2:
            	break
            if gammas[t, 0] - gammas[t+1, 0] > 0:
            	grad[t, 0] += lbd
            else:
            	if gammas[t, 0] - gammas[t+1, 0] < 0:
            		grad[t, 0] -= lbd

            if gammas[t, 1] - gammas[t+1, 1] > 0:
            	grad[t, 1] += lbd
            else:
            	if gammas[t, 1] - gammas[t+1, 1] < 0:
            		grad[t, 1] -= lbd


        gammas = gammas - grad * (0.1 / math.sqrt(i+1)) / np.linalg.norm(grad)

    return gammas




def l1_estimate_params4(V, U, lbd):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
    T = len(V)

    gammas = np.ones((T, 4))
    iters = 800
    for i in range(iters):
        grad = np.zeros((T, 4))
        for t in range(T-1):
            grad[t, 0] += V[t] * (V[t] * gammas[t, 0] - V[t+1] + gammas[t, 1] * U[t])
            grad[t, 1] += U[t] * (U[t] * gammas[t, 1] - V[t+1] + gammas[t, 0] * V[t])

            grad[t, 2] += V[t] * (V[t] * gammas[t, 2] - U[t+1] + gammas[t, 3] * U[t])
            grad[t, 3] += U[t] * (U[t] * gammas[t, 3] - U[t+1] + gammas[t, 2] * V[t])

            #if t == T-2:
            #	break
            if gammas[t, 0] - gammas[t+1, 0] > 0:
            	grad[t, 0] += lbd
            else:
            	if gammas[t, 0] - gammas[t+1, 0] < 0:
            		grad[t, 0] -= lbd

            if gammas[t, 1] - gammas[t+1, 1] > 0:
            	grad[t, 1] += lbd
            else:
            	if gammas[t, 1] - gammas[t+1, 1] < 0:
            		grad[t, 1] -= lbd


        gammas = gammas - grad * (0.1 / math.sqrt(i+1)) / np.linalg.norm(grad)

    return gammas






def delta_estimate_params2(V, U, delta):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
    T = len(V)

    gammas = np.ones((T, 2))
    iters = 300
    for i in range(iters):
        grad = np.zeros((T, 2))
        for t in range(T-1):
            grad[t, 0] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[t, 0] - V[t+1])
            grad[t, 1] += (U[t] + V[t]) * ((U[t] + V[t]) * gammas[t, 1] - U[t+1])
        gammas = gammas - grad * (0.3 / math.sqrt(i+1)) / np.linalg.norm(grad)


    return gammas


def delta_estimate_params4(V, U, delta):
    if len(V) != len(U):
        raise ValueError("Arrays must have the same size")
    T = len(V)

    gammas = np.ones((T, 4))
    iters = 100
    for i in range(iters):
        grad = np.zeros((T, 4))
        for t in range(T-1):
            grad[t, 0] += V[t] * (V[t] * gammas[t, 0] - V[t+1] + gammas[t, 1] * U[t])
            grad[t, 1] += U[t] * (U[t] * gammas[t, 1] - V[t+1] + gammas[t, 0] * V[t])

            grad[t, 2] += V[t] * (V[t] * gammas[t, 2] - U[t+1] + gammas[t, 3] * U[t])
            grad[t, 3] += U[t] * (U[t] * gammas[t, 3] - U[t+1] + gammas[t, 2] * V[t])

        gammas = gammas - grad * (0.5 / math.sqrt(i+1)) / np.linalg.norm(grad)

        # make projection onto convex set defined by delta
        x = cp.Variable((T, 4))
        expression = cp.sum_squares(x - gammas)
        constraints = [x >= 0]
        for t in range(T-1):
        	constraints.append(cp.pnorm(x[t] - x[t-1], 'inf') <= delta)
        prob = cp.Problem(cp.Minimize(expression), constraints)
        prob.solve()
        gammas = x.value

    print("done")
    return gammas
