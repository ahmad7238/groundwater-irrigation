
import numpy as np

def TDMAsolver(a, b, c, d):
    n = len(b)
    x = np.zeros(n)

    # Modify the first-row coefficient
    c[0] = c[0] / b[0]  # Division by zero risk
    d[0] = d[0] / b[0]  # Division by zero would imply a singular matrix
    for i in list(range(1, n - 1)):
        temp = b[i] - a[i] * c[i - 1]
        c[i] = c[i] / temp
        d[i] = (d[i] - a[i] * d[i - 1]) / temp

    d[n - 1] = (d[n - 1] - a[n - 1] * d[n - 2]) / (b[n - 1] - a[n - 1] * c[n - 2])
    x[n - 1] = d[n - 1]
    for i in list(range(n - 2, -1, -1)):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def VGSWC(psi, thetas, thetar, alfa, n_v):
    if psi < 0:
        m = 1 - 1 / n_v
        theta = ((thetas - thetar) / (1 + alfa * abs(psi) ** n_v) ** m) + thetar
    else:
        theta = thetas
    return theta


def k_unsat(psi, thetas, thetar, alfa, n_v, k_s):
    theta = VGSWC(psi, thetas, thetar, alfa, n_v)
    se = (theta - thetar) / (thetas - thetar)
    m = 1 - 1 / n_v
    kunsat = k_s * (se ** 0.5) * (1 - (1 - se ** (1 / m)) ** m) ** 2
    return kunsat


def Cw(psi, thetas, thetar, alfa, n_v):
    if psi < 0:
        cw = ((alfa ** n_v) * (thetas - thetar) * (n_v - 1) * ((-psi) ** (n_v - 1))) / (
                (1 + (alfa * (-psi)) ** n_v) ** (2 - (1 / n_v)))
    else:
        cw = 0
    return cw

def a_trap(b, ss, h):
    area = h * (b + h * ss)
    return area


def p_trap(b, ss, h):
    perimeter = b + 2 * h * np.sqrt(1 + ss ** 2)
    return perimeter


def t_trap(b, ss, h):
    freesurf = b + 2 * h * ss
    return freesurf


def k_l(h_c, t, alph, f, w):
    infiltration = (h_c * (t ** alph) + f * t) / w
    return infiltration

