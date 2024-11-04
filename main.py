import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
from numpy import log
from scipy.integrate import solve_ivp

LATEX_OUT_PATH = './doc/pytex'

def fn(x: float, u: float):
    return x * log(x + u) + 2 * x

def run_rk4(fn, y0, interval, tau):
    space = generate_space(interval, tau)
    y = y0
    points = []
    for tn in space:
        points += [(tn, y)]
        k1 = fn(tn, y)
        k2 = fn(tn + tau / 2, y + tau / 2 * k1)
        k3 = fn(tn + tau / 2, y + tau / 2 * k2)
        k4 = fn(tn + tau, y + tau * k3)
        y = y + tau / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return np.array(points)

def run_rk2_sigma(fn, y0, interval, tau, sigma):
    space = generate_space(interval, tau)
    y = y0
    points = []
    for tn in space:
        points += [(tn, y)]
        k1 = fn(tn, y)
        k2 = fn(tn + tau / (2 * sigma), y + tau / (2 * sigma) * k1)
        y = y + tau * ((1 - sigma) * k1 + sigma * k2)
    return np.array(points)

def evaluate_precision(solver_fn, tau, p):
    solution_tau = solver_fn(tau)
    solution_2tau = solver_fn(2 * tau)
    # print('Precision solutions: ', solution_tau, solution_2tau)
    return np.abs(solution_2tau - solution_tau) / (2 ** p - 1)
    
def generate_space(interval, step_size):
    a, b = interval
    step_count = round((b - a) / step_size) + 1
    return np.linspace(a, b, step_count)

def tex_tabulate(filename, *args, **kwargs):
    with open(os.path.join(LATEX_OUT_PATH, filename), 'w') as f:
        f.write(tabulate(*args, **kwargs))

def save_figure(name):
    path = os.path.join(LATEX_OUT_PATH, name + '.pgf')
    plt.savefig(path, format='pgf')
    plt.close()

def plot_simple(name, solutions):
    markers = ['+', 'x', '1']
    assert len(markers) >= len(solutions)
    for marker, (k, v) in zip(markers, solutions.items()):
        x, y = np.transpose(v)
        plt.plot(x, y, label=k, linewidth=1.0, marker=marker)
    plt.legend()
    save_figure(name)

def plot_differences(name, base, solutions):
    markers = ['+', 'x', '1']
    assert len(markers) >= len(solutions)
    base_x, base_y = np.transpose(base)
    for marker, (k, v) in zip(markers, solutions.items()):
        x, y = np.transpose(v)
        assert len(x) == len(base_x)
        for (x1, x2) in zip(x, base_x):
            assert abs(x1 - x2) < 1e-9
        plt.plot(x, (y - base_y), label=k, linewidth=1.0, marker=marker)
    plt.legend()
    save_figure(name)

def plot_differences_keyed(name, solutions, base_name):
    assert base_name in solutions.keys()
    plot_differences(
        name, solutions[base_name],
        {k: v for (k, v) in solutions.items() if k != base_name}
    )

def zeroify(s):
    result = [s[0]]
    for t, v in s[1:]:
        result += [((result[-1][0] + t) / 2, None)]
        result += [(t, v)]
    return result

def join_tables(*tables):
    for t in tables[1:]:
        assert len(tables[0]) == len(t)
        for (t1, _), (t2, _) in zip(tables[0], t):
            assert abs(t1 - t2) <= 1e-9

    tables = np.array(tables)
    tvals = tables[0].transpose()[0]
    vals = tables[:,:,1]
    return np.concatenate(([tvals], vals)).transpose()

if __name__ == '__main__':
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    os.makedirs(LATEX_OUT_PATH, exist_ok=True)

    tau1 = 0.1
    tau2 = 0.05
    sigma = 0.5
    y0 = 1
    interval = (0, 1)

    ivp_sol = solve_ivp(fn, interval, [y0], t_eval=generate_space((0, 1), tau2), method='RK23')
    ivp_sol = np.array(list(zip(ivp_sol.t, ivp_sol.y[0])))

    rk4_at_tau = lambda t: run_rk4(fn, y0, interval, t)
    rk2_at_tau = lambda t: run_rk2_sigma(fn, y0, interval, t, sigma)
    rk4_sol = lambda t: rk4_at_tau(t)[-1][1]
    rk2_sol = lambda t: rk2_at_tau(t)[-1][1]

    tau1_sols = {
        'RK4': rk4_at_tau(tau1),
        'RK2': rk2_at_tau(tau1)
    }

    tau2_sols = {
        'RK4': rk4_at_tau(tau2),
        'RK2': rk2_at_tau(tau2)
    }

    tau1_precision = {
        'RK4': evaluate_precision(rk4_sol, tau1, 4),
        'RK2': evaluate_precision(rk2_sol, tau1, 2)
    }

    tau2_precision = {
        'RK4': evaluate_precision(rk4_sol, tau2, 4),
        'RK2': evaluate_precision(rk2_sol, tau2, 2)
    }

    plot_simple('tau1_simple', tau1_sols | {'solve\\_ivp': ivp_sol})
    plot_simple('tau2_simple', tau2_sols | {'solve\\_ivp': ivp_sol})

    plot_differences_keyed('tau1_diff', tau1_sols, 'RK2')
    plot_differences_keyed('tau2_diff', tau2_sols, 'RK2')

    plot_differences('tau1_diff_ivp', ivp_sol[::2], tau1_sols)
    plot_differences('tau2_diff_ivp', ivp_sol, tau2_sols)

    # plot_simple('', ivp_sol, tau2_sols)

    tex_tabulate('rt4.tex', join_tables(zeroify(tau1_sols['RK4']), tau2_sols['RK4'], ivp_sol), headers=['$t_n$', '$y_n$, kai $\\tau = \\tone$', '$y_n$, kai $\\tau = \\ttwo$', '$y_n$ su \\texttt{solve\\_ivp}'], tablefmt='latex_raw', floatfmt=['g', '.8f', '.8f', '.8f'])
    tex_tabulate('rt2.tex', join_tables(zeroify(tau1_sols['RK2']), tau2_sols['RK2'], ivp_sol), headers=['$t_n$', '$y_n$, kai $\\tau = \\tone$', '$y_n$, kai $\\tau = \\ttwo$', '$y_n$ su \\texttt{solve\\_ivp}'], tablefmt='latex_raw', floatfmt=['g', '.8f', '.8f', '.8f'])
    # tex_tabulate('rt4_t2.tex', rt4_points_t2, headers=['$t_n$', '$y_n$'], tablefmt='latex_raw')
    # tex_tabulate('rt2_t1.tex', rt2_points_t1, headers=['$t_n$', '$y_n$'], tablefmt='latex_raw')
    # tex_tabulate('rt2_t2.tex', rt2_points_t2, headers=['$t_n$', '$y_n$'], tablefmt='latex_raw')
    
