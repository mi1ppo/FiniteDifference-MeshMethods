"""
Расчеты по ЛР 1. Печатает таблицы в консоль, графики кладет в ../report/figures.

Запуск из корня проекта:  .venv/Scripts/python code/run.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import problem as pb
import schemes as sc

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'report', 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120,
                     'axes.grid': True, 'grid.alpha': 0.3})


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


# отладка
def debug_quadratic():
    """На u* = x^2 схема O(h^2) должна давать погрешность ~ ошибки округления."""
    print('Отладка, u* = x^2')
    print('%8s %8s %12s' % ('n', 'схема', '||z_h||'))
    for n in (10, 20, 40):
        for order in (1, 2):
            x, v = sc.solve(n, order, f=pb.f_dbg,
                            ga=pb.gamma_a_dbg(), gb=pb.gamma_b_dbg())
            print('%8d %8s %12.3e'
                  % (n, 'O(h^%d)' % order, sc.err_c(v, pb.u_dbg(x))))


# пункт 1: график u*
def fig_exact():
    x = np.linspace(pb.A, pb.B, 3001)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, pb.u_exact(x), lw=1.6, color='#1f4e79')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u^*(x)$')
    savefig(fig, 'exact')


# пункты 2-3: численное решение на грубой сетке
def fig_compare(n=30):
    xf = np.linspace(pb.A, pb.B, 3001)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, order in zip(axes, (1, 2)):
        x, v = sc.solve(n, order)
        ax.plot(xf, pb.u_exact(xf), lw=1.4, color='#1f4e79', label='$u^*$')
        ax.plot(x, v, 'o--', ms=3.5, lw=0.9, color='#c0392b', label='$v_h$')
        ax.set_xlabel('$x$')
        ax.set_title('схема $O(h^%d)$, $n=%d$' % (order, n))
        ax.legend(fontsize=9)
    axes[0].set_ylabel('$u$')
    savefig(fig, 'compare')

    print('\nПогрешность при n = %d' % n)
    for order in (1, 2):
        x, v = sc.solve(n, order)
        print('  O(h^%d): ||z_h|| = %.3e' % (order, sc.err_c(v, pb.u_exact(x))))


# пункт 4: сходимость
# последовательность сеток: шаг каждый раз уменьшается в 2 раза
NS = [20 * 2 ** j for j in range(14)]


def steps():
    return np.array([(pb.B - pb.A) / n for n in NS])


def convergence(dtype=np.float64):
    """Погрешность обеих схем на всех сетках NS."""
    out = {1: [], 2: []}
    for n in NS:
        for order in (1, 2):
            x, v = sc.solve(n, order, dtype=dtype)
            out[order].append(sc.err_c(v, pb.u_exact(x)))
    return {k: np.array(val) for k, val in out.items()}


def orders(err):
    """Эмпирический порядок k = log2(||z_h|| / ||z_{h/2}||) по соседним сеткам."""
    return [np.nan] + [np.log2(err[j - 1] / err[j]) for j in range(1, len(err))]


def fig_convergence(err, last=None):
    """
    -lg||z_h|| от -lg h для обеих схем и эталонные прямые с наклоном k.
    last -- сколько сеток показывать (до начала влияния округления).
    """
    h = steps()[:last]
    mlh = -np.log10(h)
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {1: ('o-', '#c0392b'), 2: ('s-', '#1f4e79')}
    for order in (1, 2):
        mlz = -np.log10(err[order][:last])
        ax.plot(mlh, mlz, style[order][0], color=style[order][1], ms=4,
                label='схема $O(h^%d)$' % order)
        # эталонная прямая с наклоном k проводится через среднюю точку:
        # на последних сетках уже может сказываться округление
        # и сдвигается вниз, чтобы не сливаться с точками
        m = len(mlh) // 2
        c = mlz[m] - order * mlh[m] - 0.5
        ax.plot(mlh, order * mlh + c, '--', lw=1.0, color='gray')
        ax.annotate('$k=%d$' % order, (mlh[-1], order * mlh[-1] + c),
                    textcoords='offset points', xytext=(-10, -16), color='gray')
    ax.set_xlabel(r'$-\lg h$')
    ax.set_ylabel(r'$-\lg \|z_h\|$')
    ax.legend()
    savefig(fig, 'convergence')


def print_convergence(err):
    k1, k2 = orders(err[1]), orders(err[2])
    print('\nСходимость (двойная точность)')
    print('%8s %10s %12s %6s %12s %6s'
          % ('n', 'h', '||z|| O(h)', 'k', '||z|| O(h^2)', 'k'))
    for j, n in enumerate(NS):
        print('%8d %10.2e %12.3e %6.2f %12.3e %6.2f'
              % (n, (pb.B - pb.A) / n, err[1][j], k1[j], err[2][j], k2[j]))


# пункт 5: ошибки округления
# сетки до n ~ 1.3 млн, чтобы увидеть рост погрешности и в двойной точности
NS_LONG = [20 * 2 ** j for j in range(17)]


def convergence_long(dtype):
    out = {1: [], 2: []}
    for n in NS_LONG:
        for order in (1, 2):
            x, v = sc.solve(n, order, dtype=dtype)
            out[order].append(sc.err_c(v, pb.u_exact(np.float64(x))))
    return {k: np.array(val) for k, val in out.items()}


def fig_roundoff(e64, e32):
    mlh = -np.log10(np.array([(pb.B - pb.A) / n for n in NS_LONG]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, order in zip(axes, (1, 2)):
        ax.plot(mlh, -np.log10(e64[order]), 'o-', ms=4, color='#1f4e79',
                label='двойная (float64)')
        ax.plot(mlh, -np.log10(e32[order]), 's-', ms=4, color='#c0392b',
                label='одинарная (float32)')
        for e, col in ((e64, '#1f4e79'), (e32, '#c0392b')):
            j = int(np.argmin(e[order]))
            ax.plot(mlh[j], -np.log10(e[order][j]), 'o', ms=11, mfc='none',
                    mec=col, mew=1.5)
        ax.set_xlabel(r'$-\lg h$')
        ax.set_title('схема $O(h^%d)$' % order)
        ax.legend(fontsize=9, loc='upper left')
    axes[0].set_ylabel(r'$-\lg \|z_h\|$')
    savefig(fig, 'roundoff')


def print_roundoff(e64, e32):
    print('\nОдинарная и двойная точность')
    print('%8s %12s %12s %12s %12s'
          % ('n', 'O(h) dbl', 'O(h) sgl', 'O(h^2) dbl', 'O(h^2) sgl'))
    for j, n in enumerate(NS_LONG):
        print('%8d %12.3e %12.3e %12.3e %12.3e'
              % (n, e64[1][j], e32[1][j], e64[2][j], e32[2][j]))
    print('\nМинимум погрешности (дальше она растет из-за округления)')
    for name, e in (('float32', e32), ('float64', e64)):
        for order in (1, 2):
            j = int(np.argmin(e[order]))
            edge = ' (последняя сетка, рост не достигнут)' if j == len(NS_LONG) - 1 else ''
            print('  %s, O(h^%d): n = %7d, ||z|| = %.2e%s'
                  % (name, order, NS_LONG[j], e[order][j], edge))


# пункт 6: число разбиений для заданной точности
N_MAX = 2 ** 21  # ~2.1 млн разбиений


def min_nodes(order, eps, n_max=N_MAX):
    """
    Наименьшее n, при котором ||z_h|| <= eps (двойная точность).

    Сначала n удваивается, пока точность не достигнута, затем
    нужное n уточняется двоичным поиском между n/2 и n.
    """
    def err(n):
        x, v = sc.solve(n, order)
        return sc.err_c(v, pb.u_exact(x))

    hi = 2
    while err(hi) > eps:
        if hi * 2 > n_max:
            return None, None
        hi *= 2
    lo = hi // 2  # err(lo) > eps >= err(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi, err(hi)


def print_min_nodes():
    print('\nЧисло разбиений для заданной точности (двойная точность)')
    print('%8s %10s %12s %10s %12s'
          % ('eps', 'n, O(h)', '||z||', 'n, O(h^2)', '||z||'))
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        n1, z1 = min_nodes(1, eps)
        n2, z2 = min_nodes(2, eps)
        print('%8.0e %10s %12s %10s %12s'
              % (eps,
                 n1 if n1 else '> 2*10^6', '%.3e' % z1 if n1 else '-',
                 n2 if n2 else '> 2*10^6', '%.3e' % z2 if n2 else '-'))


# задание на "5": метод Рунге
def runge(order, ns):
    """
    Оценка погрешности по Рунге и уточнение решения.

    Для каждой пары сеток (h, h/2) в общих узлах:
        R = (v_{h/2} - v_h) / (2^k - 1)    -- оценка u - v_{h/2},
        v_уточн = v_{h/2} + R               -- уточненное решение.
    Возвращает строки (n, ||R||, ||u - v_{h/2}||, ||u - v_уточн||).
    """
    k = order
    sols = {n: sc.solve(n, order) for n in ns}
    rows = []
    for n in ns[:-1]:
        x, v1 = sols[n]
        v2 = sols[2 * n][1][::2]  # решение на сетке h/2 в общих узлах
        R = (v2 - v1) / (2 ** k - 1)
        u = pb.u_exact(x)
        rows.append((n,
                     np.max(np.abs(R)),
                     np.max(np.abs(u - v2)),
                     np.max(np.abs(u - (v2 + R)))))
    return rows


def print_runge(order, rows):
    print('\nМетод Рунге, схема O(h^%d)' % order)
    print('%8s %12s %12s %8s %12s %8s'
          % ('n', '||R||', '||z_h/2||', 'R/z', 'уточн.', 'порядок'))
    prev = None
    for n, R, z2, zr in rows:
        k = '-' if prev is None else '%.2f' % np.log2(prev / zr)
        print('%8d %12.3e %12.3e %8.3f %12.3e %8s' % (n, R, z2, R / z2, zr, k))
        prev = zr


def fig_runge(rows1, rows2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, rows, order in ((axes[0], rows1, 1), (axes[1], rows2, 2)):
        h = (pb.B - pb.A) / np.array([r[0] for r in rows])
        mlh = -np.log10(h)
        z2 = np.array([r[2] for r in rows])
        zr = np.array([r[3] for r in rows])
        ax.plot(mlh, -np.log10(z2), 'o-', ms=4, color='#1f4e79',
                label=r'$v_{h/2}$ (без уточнения)')
        ax.plot(mlh, -np.log10(zr), 's-', ms=4, color='#c0392b',
                label=r'$v_{h/2} + R$ (уточненное)')
        # эталонные прямые: порядок схемы и порядок уточненного решения,
        # найденный по таблице (2 для схемы O(h), 4 для схемы O(h^2))
        m = len(mlh) // 2
        for kk, base in ((order, z2), (2 * order, zr)):
            c = -np.log10(base[m]) - kk * mlh[m] - 0.5
            ax.plot(mlh, kk * mlh + c, '--', lw=0.9, color='gray')
            ax.annotate('$k=%d$' % kk, (mlh[-1], kk * mlh[-1] + c),
                        textcoords='offset points', xytext=(-10, -14),
                        color='gray', fontsize=9)
        ax.set_xlabel(r'$-\lg h$')
        ax.set_ylabel(r'$-\lg \|z\|$')
        ax.set_title('схема $O(h^%d)$' % order)
        ax.legend(fontsize=9, loc='upper left')
    savefig(fig, 'runge')


PARTS = ('base', 'conv', 'round', 'nodes', 'runge')


def main(parts=PARTS):
    if 'base' in parts:
        debug_quadratic()
        fig_exact()
        fig_compare()

    # пункт 4
    if 'conv' in parts:
        e64 = convergence(np.float64)
        print_convergence(e64)
        fig_convergence(e64)

    # пункт 5
    if 'round' in parts:
        l64 = convergence_long(np.float64)
        l32 = convergence_long(np.float32)
        print_roundoff(l64, l32)
        fig_roundoff(l64, l32)

    # пункт 6
    if 'nodes' in parts:
        print_min_nodes()

    # задание на "5"
    if 'runge' in parts:
        r1 = runge(1, [20 * 2 ** j for j in range(12)])
        r2 = runge(2, [20 * 2 ** j for j in range(10)])
        print_runge(1, r1)
        print_runge(2, r2)
        fig_runge(r1, r2)

    print('\nГрафики:', os.path.normpath(FIG))


if __name__ == '__main__':
    # можно запустить только часть расчетов: python code/run.py runge
    main(sys.argv[1:] or PARTS)
