from numpy import log

def fn(x: float, u: float):
    return x * log(x + u) + 2 * x

def run_rt4(fn, tau: float, step_count: int):
    t = 0
    y = 0
    for _ in range(step_count):
        k1 = fn(t, y)
        k2 = fn(t + tau / 2, y + tau / 2 * k1)
        k3 = fn(t + tau / 2, y + tau / 2 * k2)
        k4 = fn(t + tau, y + tau * k3)
        y = y + tau / 6 * (k1 + 2*k2 + 2*k3 + k4)
    

