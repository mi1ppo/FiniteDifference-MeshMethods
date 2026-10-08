"""
Разностные схемы для краевой задачи третьего рода и метод прогонки.

Внутренние узлы (общие для обеих схем, порядок O(h^2)):

    -(y_{i-1} - 2 y_i + y_{i+1}) / h^2 + p_i (y_{i+1} - y_{i-1}) / (2h) + q_i y_i = f_i

Схемы отличаются только аппроксимацией u' в краевых условиях.
"""

import numpy as np

import problem as pb


def thomas(lo, di, up, rhs, dtype=np.float64):
    """
    Метод прогонки для трехдиагональной системы.

    lo[i] -- коэффициент при y[i-1] в строке i (lo[0] не используется),
    di[i] -- коэффициент при y[i],
    up[i] -- коэффициент при y[i+1] в строке i (up[-1] не используется).
    """
    a = np.array(lo, dtype=dtype)
    b = np.array(di, dtype=dtype)
    c = np.array(up, dtype=dtype)
    d = np.array(rhs, dtype=dtype)
    n = b.size

    # прямой ход: исключаем поддиагональ
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # обратный ход
    y = np.empty(n, dtype=dtype)
    y[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]
    return y


def assemble(n, order, dtype=np.float64, f=None, ga=None, gb=None):
    """
    Собрать трехдиагональную систему на сетке из n+1 узла.

    order = 1 -- u' на концах заменяется односторонней разностью, схема O(h);
    order = 2 -- u' на концах уточняется с помощью уравнения, схема O(h^2).

    По умолчанию берется тестовая задача с решением u*; аргументы
    f, ga, gb позволяют подставить отладочную задачу.
    """
    if f is None:
        f, ga, gb = pb.f_star, pb.gamma_a_star(), pb.gamma_b_star()

    h = dtype((pb.B - pb.A) / n)
    x = np.linspace(pb.A, pb.B, n + 1).astype(dtype)
    pv = pb.p(x).astype(dtype)
    qv = pb.q(x).astype(dtype)
    fv = np.asarray(f(x), dtype=dtype)

    lo = np.zeros(n + 1, dtype=dtype)
    di = np.zeros(n + 1, dtype=dtype)
    up = np.zeros(n + 1, dtype=dtype)
    rhs = np.zeros(n + 1, dtype=dtype)

    # внутренние узлы i = 1..n-1
    i = np.arange(1, n)
    lo[i] = -1 / h ** 2 - pv[i] / (2 * h)
    di[i] = 2 / h ** 2 + qv[i]
    up[i] = -1 / h ** 2 + pv[i] / (2 * h)
    rhs[i] = fv[i]

    aa, ba = dtype(pb.ALPHA_A), dtype(pb.BETA_A)
    ab, bb = dtype(pb.ALPHA_B), dtype(pb.BETA_B)

    if order == 1:
        # -alpha_a (y_1 - y_0)/h + beta_a y_0 = gamma_a
        di[0] = aa / h + ba
        up[0] = -aa / h
        rhs[0] = ga
        #  alpha_b (y_n - y_{n-1})/h + beta_b y_n = gamma_b
        lo[n] = -ab / h
        di[n] = ab / h + bb
        rhs[n] = gb

    elif order == 2:
        # u'(a) ~ [ (y_1 - y_0)/h - (h/2)(q_0 y_0 - f_0) ] / Da,  Da = 1 + h p_0 / 2
        # подставляем в условие и умножаем строку на Da
        Da = 1 + h * pv[0] / 2
        di[0] = aa / h + aa * h * qv[0] / 2 + ba * Da
        up[0] = -aa / h
        rhs[0] = ga * Da + aa * h * fv[0] / 2
        # u'(b) ~ [ (y_n - y_{n-1})/h + (h/2)(q_n y_n - f_n) ] / Db,  Db = 1 - h p_n / 2
        Db = 1 - h * pv[n] / 2
        lo[n] = -ab / h
        di[n] = ab / h + ab * h * qv[n] / 2 + bb * Db
        rhs[n] = gb * Db + ab * h * fv[n] / 2

    else:
        raise ValueError('order должен быть 1 или 2')

    return x, lo, di, up, rhs


def solve(n, order, dtype=np.float64, f=None, ga=None, gb=None):
    """Решить разностную задачу, вернуть узлы x и решение v."""
    x, lo, di, up, rhs = assemble(n, order, dtype, f, ga, gb)
    return x, thomas(lo, di, up, rhs, dtype=dtype)


def err_c(v, u):
    """Норма погрешности ||z_h|| = max_i |u_i - v_i| (считается в float64)."""
    return float(np.max(np.abs(np.asarray(u, dtype=np.float64)
                               - np.asarray(v, dtype=np.float64))))
