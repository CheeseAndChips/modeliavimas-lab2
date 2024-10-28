import numpy as np
from numpy import log
from scipy.integrate import solve_ivp

def fn(x: float, u: float):
    return x * log(x + u) + 2 * x

def run_rt4(fn, y0, interval, tau):
    space = generate_space(interval, tau)
    y = y0
    points = [(space[0], y)]
    for tn in space[1:]:
        k1 = fn(tn, y)
        k2 = fn(tn + tau / 2, y + tau / 2 * k1)
        k3 = fn(tn + tau / 2, y + tau / 2 * k2)
        k4 = fn(tn + tau, y + tau * k3)
        y = y + tau / 6 * (k1 + 2*k2 + 2*k3 + k4)
        points += [(tn, y)]
    return points

def run_rt2_sigma(fn, y0, interval, tau, sigma):
    space = generate_space(interval, tau)
    y = y0
    points = [(space[0], y)]
    for tn in space[1:]:
        k1 = fn(tn, y)
        k2 = fn(tn + tau / (2 * sigma), y + tau / (2 * sigma) * k1)
        y = y + tau * ((1 - sigma) * k1 + sigma * k2)
        points += [(tn, y)]
    return points

def evaluate_precision(solver_fn, tau, p):
    solution_tau = solver_fn(tau)
    solution_2tau = solver_fn(2 * tau)
    print(solution_tau, solution_2tau)
    return np.abs(solution_2tau - solution_tau) / (2 ** p - 1)
    
def generate_space(interval, step_size):
    a, b = interval
    step_count = round((b - a) / step_size) + 1
    return np.linspace(a, b, step_count)

if __name__ == '__main__':
    tau = 0.05
    sigma = 1
    y0 = 1
    interval = (0, 1)

    rt4_points = run_rt4(fn, y0, interval, tau)
    rt2_points = run_rt2_sigma(fn, y0, interval, tau, sigma)
    ivp_points = solve_ivp(fn, interval, [y0], method='RK23')

    print('Solutions:')
    print(rt4_points[-1])
    print(rt2_points[-1])
    print(ivp_points.y[0][-1], ivp_points.t[-1])
    
    print('Precision evaluation:')
    print(f'RT4, tau = {tau}:', evaluate_precision(lambda t: run_rt4(fn, y0, interval, t)[-1][1], tau, 4))
    print(f'RT2, tau = {tau}:', evaluate_precision(lambda t: run_rt2_sigma(fn, y0, interval, t, sigma)[-1][1], tau, 2))

