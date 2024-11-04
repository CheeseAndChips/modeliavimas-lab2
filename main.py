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

class PrecisionEval:
    def __init__(self, val, tex):
        self.val = val
        self.tex = tex

def evaluate_precision(solver_fn, tau, p):
    solution_tau = solver_fn(tau)
    solution_2tau = solver_fn(2 * tau)
    def tex_approx_abs_error(y2t, yt, p):
        return f'\\frac{{|{y2t} - {yt}|}}{{2^{p} - 1}}'

    print(f'tau: {solution_2tau}, tau: {solution_tau}')
    val = np.abs(solution_2tau - solution_tau) / (2 ** p - 1)
    return PrecisionEval(val, tex_approx_abs_error(solution_2tau, solution_tau, p))
    
def generate_space(interval, step_size):
    a, b = interval
    step_count = round((b - a) / step_size) + 1
    return np.linspace(a, b, step_count)

def tex_tabulate(filename, *args, **kwargs):
    with open(os.path.join(LATEX_OUT_PATH, filename), 'w') as f:
        f.write(tabulate(*args, **kwargs))

def save_figure(name):
    path = os.path.join(LATEX_OUT_PATH, name + '.pgf')
    plt.tight_layout()
    plt.savefig(path, format='pgf')
    plt.close()

def plot_simple(name, solutions):
    markers = ['+', 'x', '1']
    assert len(markers) >= len(solutions)
    for marker, (k, v) in zip(markers, solutions.items()):
        x, y = np.transpose(v)
        plt.plot(x, y, label=k, linewidth=1.0, marker=marker)
    plt.xlabel('$t_n$')
    plt.ylabel('$y_n$')
    plt.legend()
    save_figure(name)

def plot_differences(name, base, solutions, divide_base=None):
    markers = ['+', 'x', '1']
    assert len(markers) >= len(solutions)
    base_x, base_y = np.transpose(base)
    if not divide_base:
        divide_base = [False for _ in solutions]
    for marker, (k, v), divide in zip(markers, solutions.items(), divide_base):
        x, y = np.transpose(v)
        if divide:
            base_x_it, base_y_it = base_x[::2], base_y[::2]
        else:
            base_x_it, base_y_it = base_x, base_y
        assert len(x) == len(base_x_it)
        for (x1, x2) in zip(x, base_x_it):
            assert abs(x1 - x2) < 1e-9
        plt.plot(x, (y - base_y_it), label=k, linewidth=1.0, marker=marker)
    plt.xlabel('$t_n$')
    plt.ylabel('$\\Delta(t)$')
    if len(solutions) > 1:
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

# TODO: labels for charts
# TODO: add code to appendix
if __name__ == '__main__':
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.figsize': (6.2, 4), # default (6.4, 4.8)
    })
    os.makedirs(LATEX_OUT_PATH, exist_ok=True)

    tau1 = 0.1
    tau2 = 0.05
    sigma = 0.5
    y0 = 1
    interval = (0, 1)

    ivp_sol = solve_ivp(fn, interval, [y0], t_eval=generate_space((0, 1), tau2), method='RK23')
    ivp_sol = np.array(list(zip(ivp_sol.t, ivp_sol.y[0])))
    ivp_sol_dict = {'$\\texttt{solve\\_ivp}$': ivp_sol}

    rk4_at_tau = lambda t: run_rk4(fn, y0, interval, t)
    rk2_at_tau = lambda t: run_rk2_sigma(fn, y0, interval, t, sigma)
    rk4_sol = lambda t: rk4_at_tau(t)[-1][1]
    rk2_sol = lambda t: rk2_at_tau(t)[-1][1]

    tau1_fmt = '$\\tau = {}$'.format(tau1)
    tau2_fmt = '$\\tau = {}$'.format(tau2)

    tau1_sols = {
        'RK4': rk4_at_tau(tau1),
        'RK2': rk2_at_tau(tau1)
    }

    tau2_sols = {
        'RK4': rk4_at_tau(tau2),
        'RK2': rk2_at_tau(tau2)
    }

    rk4_sols = {
        tau1_fmt: tau1_sols['RK4'],
        tau2_fmt: tau2_sols['RK4'],
    }

    rk2_sols = {
        tau1_fmt: tau1_sols['RK2'],
        tau2_fmt: tau2_sols['RK2'],
    }

    rk4_sols_norm = {
        tau1_fmt: rk4_sols[tau1_fmt],
        tau2_fmt: rk4_sols[tau2_fmt][::2]
    }

    rk2_sols_norm = {
        tau1_fmt: rk2_sols[tau1_fmt],
        tau2_fmt: rk2_sols[tau2_fmt][::2]
    }

    tau1_precision = {
        'RK4': evaluate_precision(rk4_sol, tau1, 4),
        'RK2': evaluate_precision(rk2_sol, tau1, 2)
    }

    tau2_precision = {
        'RK4': evaluate_precision(rk4_sol, tau2, 4),
        'RK2': evaluate_precision(rk2_sol, tau2, 2)
    }

    plot_simple('rk4_simple', rk4_sols | ivp_sol_dict)
    plot_simple('rk2_simple', rk2_sols | ivp_sol_dict)

    plot_differences_keyed('rk4_diff', rk4_sols_norm, tau1_fmt)
    plot_differences_keyed('rk2_diff', rk2_sols_norm, tau1_fmt)

    plot_differences('rk4_diff_ivp', ivp_sol, rk4_sols, divide_base=[True, False])
    plot_differences('rk2_diff_ivp', ivp_sol, rk2_sols, divide_base=[True, False])

    plot_simple('tau1_simple', tau1_sols | ivp_sol_dict)
    plot_simple('tau2_simple', tau2_sols | ivp_sol_dict)

    plot_differences_keyed('tau1_diff', tau1_sols, 'RK2')
    plot_differences_keyed('tau2_diff', tau2_sols, 'RK2')

    def format_float(f):
        regular_string = str(f)
        if 'e' not in regular_string:
            return regular_string
        base, exponent = regular_string.split('e')
        exponent = int(exponent)
        return f'{base} \\cdot 10^{{{exponent}}}'

    with open(os.path.join(LATEX_OUT_PATH, 'rk4_precision.tex'), 'w') as f:
        f.write(f'{tau2_precision["RK4"].tex} \\approx {format_float(tau2_precision["RK4"].val)}')

    with open(os.path.join(LATEX_OUT_PATH, 'rk2_precision.tex'), 'w') as f:
        f.write(f'{tau2_precision["RK2"].tex} \\approx {format_float(tau2_precision["RK2"].val)}')

    tex_tabulate('rt4.tex', join_tables(zeroify(tau1_sols['RK4']), tau2_sols['RK4'], ivp_sol), headers=['$t_n$', '$y_n$, kai $\\tau = \\tone$', '$y_n$, kai $\\tau = \\ttwo$', '$y_n$ su \\texttt{solve\\_ivp}'], tablefmt='latex_raw', floatfmt=['g', '.8f', '.8f', '.8f'])
    tex_tabulate('rt2.tex', join_tables(zeroify(tau1_sols['RK2']), tau2_sols['RK2'], ivp_sol), headers=['$t_n$', '$y_n$, kai $\\tau = \\tone$', '$y_n$, kai $\\tau = \\ttwo$', '$y_n$ su \\texttt{solve\\_ivp}'], tablefmt='latex_raw', floatfmt=['g', '.8f', '.8f', '.8f'])

