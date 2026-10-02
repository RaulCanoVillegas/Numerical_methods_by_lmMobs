"""
matrices.py -- Las seis matrices de prueba de la tarea (Test Matrix Toolbox, Higham).

Todas devuelven un arreglo float de n x n. Indices matematicos desde 1 en las
formulas; en el codigo, desde 0.

    cauchy(n)              a_ij = 1/(x_i + y_j),  x_i = y_i = i
    chebvand(n)            a_ij = T_{i-1}(p_j),   p = n puntos equiespaciados en [0,1]
    forsythe(n, alpha, lam) bloque de Jordan perturbado, a_n1 = alpha
    hilb(n)                a_ij = 1/(i + j - 1)
    kahan(n, theta, pert)  triangular superior (trapezoidal), con perturbacion diagonal
    moler(n, alpha)        U^T U con U = triw(n, alpha)

sistema(A) devuelve (A, b) con b = A @ ones, de modo que la solucion exacta es 1.
"""
import numpy as np

EPS = np.finfo(float).eps


def cauchy(n):
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + i[None, :])


def chebvand(n, p=None):
    """C[i, j] = T_i(p_j), i = 0..n-1 (grado i), con la recurrencia de tres terminos.
    Por defecto p = n puntos equiespaciados en [0, 1] (valor por defecto de la toolbox)."""
    if p is None:
        p = np.linspace(0.0, 1.0, n)
    p = np.asarray(p, dtype=float)
    C = np.empty((n, len(p)))
    C[0, :] = 1.0
    if n > 1:
        C[1, :] = p
    for k in range(1, n - 1):
        C[k + 1, :] = 2.0 * p * C[k, :] - C[k - 1, :]
    return C


def forsythe(n, alpha=np.sqrt(EPS), lam=0.0):
    A = lam * np.eye(n) + np.diag(np.ones(n - 1), 1)
    A[n - 1, 0] = alpha
    return A


def hilb(n):
    i = np.arange(1, n + 1, dtype=float)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def kahan(n, theta=1.2, pert=25.0):
    s, c = np.sin(theta), np.cos(theta)
    A = np.diag(s ** np.arange(n)) @ (np.eye(n) - c * np.triu(np.ones((n, n)), 1))
    A += pert * EPS * np.diag(np.arange(n, 0, -1, dtype=float))
    return A


def moler(n, alpha=-1.0):
    U = np.eye(n) + alpha * np.triu(np.ones((n, n)), 1)     # triw(n, alpha)
    return U.T @ U


def sistema(A):
    """Devuelve (A, b) con b = A @ ones(n): la solucion exacta es el vector de unos."""
    return A, A @ np.ones(A.shape[0])


MATRICES = {"cauchy": cauchy, "chebvand": chebvand, "forsythe": forsythe,
            "hilb": hilb, "kahan": kahan, "moler": moler}


# ---------------------------------------------------------------------------
# Verificaciones: python matrices.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def ok(cond):
        return "OK" if cond else "REVISAR"

    # moler: forma cerrada a_ii = i, a_ij = min(i,j) - 2
    n = 8
    i = np.arange(1, n + 1)
    Mc = np.minimum(i[:, None], i[None, :]) - 2.0
    Mc[np.arange(n), np.arange(n)] = i
    print("moler = forma cerrada          :", ok(np.allclose(moler(n), Mc)))
    print("moler simetrica                :", ok(np.allclose(moler(n), moler(n).T)))

    # forsythe: a11 = 0 y cond_2 = 1/alpha
    F = forsythe(30)
    print("forsythe a11 = 0               :", ok(F[0, 0] == 0.0))
    print("forsythe cond2 = 1/alpha       :", ok(np.isclose(np.linalg.cond(F), 1 / np.sqrt(EPS), rtol=1e-6)),
          "(%.3e)" % np.linalg.cond(F))

    # hilb y cauchy: simetricas; cond(10) de referencia
    print("hilb cond(10) ~ 1.6e13         :", ok(np.isclose(np.linalg.cond(hilb(10)), 1.6e13, rtol=0.05)),
          "(%.3e)" % np.linalg.cond(hilb(10)))
    print("cauchy cond(10) ~ 6.2e13       :", ok(np.isclose(np.linalg.cond(cauchy(10)), 6.2e13, rtol=0.05)),
          "(%.3e)" % np.linalg.cond(cauchy(10)))
    print("cauchy = 1/(i+j), simetrica    :", ok(np.allclose(cauchy(6), cauchy(6).T) and cauchy(6)[0, 0] == 0.5))

    # kahan: triangular superior; cond(10) ~ 29.9 (con pert = 25)
    K = kahan(10)
    print("kahan triangular superior      :", ok(np.allclose(K, np.triu(K))))
    print("kahan cond(10) ~ 29.9          :", ok(np.isclose(np.linalg.cond(K), 29.85, rtol=0.01)),
          "(%.3e)" % np.linalg.cond(K))

    # chebvand: primera fila de unos, segunda = puntos, y contra numpy
    C = chebvand(9)
    p = np.linspace(0, 1, 9)
    print("chebvand fila 1 = 1, fila 2 = p:", ok(np.allclose(C[0], 1) and np.allclose(C[1], p)))
    print("chebvand = numpy chebvander    :",
          ok(np.allclose(C, np.polynomial.chebyshev.chebvander(p, 8).T)))

    # sistema: b = A @ 1
    A, b = sistema(hilb(5))
    print("sistema: b = A @ ones          :", ok(np.allclose(b, hilb(5).sum(axis=1))))
