"""
iterativos.py -- Jacobi, Seidel y Descenso profundo.

Los tres reciben (A, b, x0, kmax, tol) y devuelven (x, k, X):
    x : ultima iterada
    k : numero de iteraciones realizadas
    X : arreglo (k+1) x n con todas las iteradas (X[0] = x0)

Criterios de paro: ||r_k|| <= tol*||r_0||, k = kmax, o divergencia (||r_k|| >= 1e10*||r_0||).
"""
import numpy as np


def jacobi(A, b, x0, kmax, tol=1e-9):
    # Cada componente usa solo la iterada anterior: x_new = (b - (L+U) x) / d
    d = np.diag(A)
    x = x0.copy()
    r = b - A @ x
    r0 = np.linalg.norm(r)
    X = [x]
    k = 0
    while tol * r0 < np.linalg.norm(r) < 1e10 * r0 and k < kmax:
        x = (b - A @ x + d * x) / d          # A@x - d*x = (L+U) x
        r = b - A @ x
        X.append(x)
        k += 1
    return x, k, np.array(X)


def seidel(A, b, x0, kmax, tol=1e-9):
    # Cada componente usa ya las componentes nuevas de esta misma iteracion.
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    r0 = np.linalg.norm(r)
    X = [x.copy()]
    k = 0
    while tol * r0 < np.linalg.norm(r) < 1e10 * r0 and k < kmax:
        for i in range(n):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
        r = b - A @ x
        X.append(x.copy())
        k += 1
    return x, k, np.array(X)


def descenso(A, b, x0, kmax, tol=1e-9):
    # Avanza en la direccion del residuo con el paso que minimiza la energia.
    x = x0.copy()
    r = b - A @ x
    r0 = np.linalg.norm(r)
    X = [x]
    k = 0
    while tol * r0 < np.linalg.norm(r) < 1e10 * r0 and k < kmax:
        alpha = (r @ r) / (r @ (A @ r))
        x = x + alpha * r
        r = b - A @ x
        X.append(x)
        k += 1
    return x, k, np.array(X)
