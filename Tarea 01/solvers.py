"""
solvers.py -- Solvers "caseros" para la tarea de matrices de prueba.

Basado en el codigo del curso (LU, LUsol, chole, cholsol), con estos cambios:
  * LUpiv / LUpivsol: LU CON pivoteo parcial (el LU original no pivotea y
    falla con forsythe, donde a11 = 0).
  * chole: si aparece un pivote no positivo lanza NoDefinidaPositiva en lugar
    de imprimir un aviso y seguir con datos basura.
  * backsub: sustitucion hacia atras para matrices triangulares superiores
    (kahan).
  * resolver_*: envoltorios que NO modifican A ni b (las funciones originales
    trabajan "in place" y destruyen sus argumentos).
"""
import math
import numpy as np


class NoDefinidaPositiva(Exception):
    """Se lanza cuando Cholesky encuentra un pivote no positivo."""


class MatrizSingular(Exception):
    """Se lanza cuando LU encuentra una columna sin pivote no nulo."""


# ---------------------------------------------------------------------------
# LU sin pivoteo (version original del curso, se conserva para comparar)
# ---------------------------------------------------------------------------
def LU(a):
    n = len(a)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                a[i, k] = lam
    return a


def LUsol(a, b):
    n = len(a)
    for k in range(1, n):
        b[k] = b[k] - np.dot(a[k, 0:k], b[0:k])
    b[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b


# ---------------------------------------------------------------------------
# LU con pivoteo parcial: PA = LU
# ---------------------------------------------------------------------------
def LUpiv(a):
    """Factoriza 'a' in place (L sin la diagonal unitaria debajo, U arriba).
    Devuelve (a, seq), donde seq es la permutacion de filas: (P b) = b[seq]."""
    n = len(a)
    seq = np.arange(n)
    for k in range(n - 1):
        p = int(np.argmax(np.abs(a[k:n, k]))) + k     # fila del pivote
        if a[p, k] == 0.0:
            raise MatrizSingular(f"columna {k} sin pivote no nulo")
        if p != k:
            a[[k, p], :] = a[[p, k], :]               # intercambia filas completas
            seq[[k, p]] = seq[[p, k]]
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                a[i, k] = lam
    if a[n - 1, n - 1] == 0.0:
        raise MatrizSingular("pivote nulo en la ultima posicion")
    return a, seq


def LUpivsol(a, b, seq):
    """Resuelve usando la salida de LUpiv. Modifica b (se permuta primero)."""
    n = len(a)
    b = b[seq]                                        # b <- P b (copia)
    for k in range(1, n):                             # L y = P b (L unitaria)
        b[k] = b[k] - np.dot(a[k, 0:k], b[0:k])
    b[n - 1] = b[n - 1] / a[n - 1, n - 1]             # U x = y
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b


# ---------------------------------------------------------------------------
# Cholesky: A = L L^T (L triangular inferior)
# ---------------------------------------------------------------------------
def chole(a):
    n = len(a)
    for k in range(n):
        d = a[k, k] - np.dot(a[k, 0:k], a[k, 0:k])
        if not d > 0.0:                               # tambien atrapa nan
            raise NoDefinidaPositiva(f"pivote no positivo en k={k}: {d:.3e}")
        a[k, k] = math.sqrt(d)
        for i in range(k + 1, n):
            a[i, k] = (a[i, k] - np.dot(a[i, 0:k], a[k, 0:k])) / a[k, k]
    for k in range(1, n):
        a[0:k, k] = 0.0
    return a


def cholsol(L, b):
    n = len(b)
    for k in range(n):                                # L y = b
        b[k] = (b[k] - np.dot(L[k, 0:k], b[0:k])) / L[k, k]
    for k in range(n - 1, -1, -1):                    # L^T x = y
        b[k] = (b[k] - np.dot(L[k + 1:n, k], b[k + 1:n])) / L[k, k]
    return b


# ---------------------------------------------------------------------------
# Sustitucion hacia atras (matriz triangular superior)
# ---------------------------------------------------------------------------
def backsub(U, b):
    n = len(b)
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(U[k, k + 1:n], b[k + 1:n])) / U[k, k]
    return b


# ---------------------------------------------------------------------------
# Envoltorios que no destruyen A ni b
# ---------------------------------------------------------------------------
def resolver_LU(A, b):
    a, seq = LUpiv(np.array(A, dtype=float))
    return LUpivsol(a, np.array(b, dtype=float), seq)


def resolver_chol(A, b):
    L = chole(np.array(A, dtype=float))
    return cholsol(L, np.array(b, dtype=float))


def resolver_tri(U, b):
    return backsub(np.array(U, dtype=float), np.array(b, dtype=float))


# ---------------------------------------------------------------------------
# Pruebas rapidas: python solvers.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) Ejemplo del curso (LU sin pivoteo): debe dar [1, 1, 1]
    A = np.array([[3.0, -1.0, 4.0], [-2.0, 0.0, 5.0], [7.0, 2.0, -2.0]])
    print("LU del curso :", LUsol(LU(A.copy()), A @ np.ones(3)))

    # 2) LU con pivoteo en una matriz con a11 = 0 (como forsythe)
    B = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1e-8, 0.0, 0.0]])
    print("LU sin pivoteo en B (esperado: falla):",
          LUsol(LU(B.copy()), B @ np.ones(3)))
    print("LU con pivoteo en B:", resolver_LU(B, B @ np.ones(3)))

    # 3) LU con pivoteo vs numpy en una matriz aleatoria
    M = rng.standard_normal((50, 50))
    b = M @ np.ones(50)
    print("LUpiv aleatoria, error:", np.linalg.norm(resolver_LU(M, b) - 1))

    # 4) Cholesky vs numpy en una matriz SPD bien condicionada
    S = M @ M.T + 50 * np.eye(50)
    b = S @ np.ones(50)
    print("Cholesky SPD, error   :", np.linalg.norm(resolver_chol(S, b) - 1))
    print("   L propia == numpy  :", np.allclose(chole(S.copy()), np.linalg.cholesky(S)))

    # 5) Cholesky debe fallar limpiamente con Hilbert grande
    i = np.arange(1, 31)
    H = 1.0 / (i[:, None] + i[None, :] - 1)
    try:
        resolver_chol(H, H @ np.ones(30))
    except NoDefinidaPositiva as e:
        print("Hilbert n=30 -> NoDefinidaPositiva:", e)

    # 6) Sustitucion hacia atras
    U = np.triu(rng.standard_normal((20, 20))) + 5 * np.eye(20)
    print("backsub, error        :", np.linalg.norm(resolver_tri(U, U @ np.ones(20)) - 1))
