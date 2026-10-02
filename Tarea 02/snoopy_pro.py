"""
Interpolación de la figura de Snoopy (Tarea 2, Métodos Numéricos).
Métodos: Vandermonde, Lagrange y spline cúbico sujeto (programado a mano).

Uso:  python snoopy.py
Requiere: numpy, scipy, matplotlib.
Genera, junto al script:
    figuras/  -> curva1.pdf, curva2.pdf, curva3.pdf, silueta_*.pdf
    tablas/   -> metricas_curva1.tex, metricas_curva2.tex, metricas_curva3.tex, resumen.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

os.makedirs("figuras", exist_ok=True)
os.makedirs("tablas", exist_ok=True)

# Datos de la tabla: x, f(x), f'(x) en el primer y último nodo
curvas = {
    "Curva 1": dict(x=[1, 2, 5, 6, 7, 8, 10, 13, 17],
                    y=[3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5],
                    d0=1.0, dn=-0.67),
    "Curva 2": dict(x=[17, 20, 23, 24, 25, 27, 27.7],
                    y=[4.5, 7.0, 6.1, 5.6, 5.8, 5.2, 4.1],
                    d0=3.0, dn=-4.0),
    "Curva 3": dict(x=[27.7, 28, 29, 30],
                    y=[4.1, 4.3, 4.1, 3.0],
                    d0=0.33, dn=-1.5),
}
NMALLA = 400
YLIM = (0, 9)

# ---------- 1) Vandermonde ----------
def vandermonde(x, y):
    x = np.array(x, float); y = np.array(y, float)
    V = np.vander(x, increasing=True)          # columnas 1, x, x^2, ...
    a = np.linalg.solve(V, y)                  # coeficientes a0..an
    return (lambda t: np.polynomial.polynomial.polyval(t, a)), a, np.linalg.cond(V)

# ---------- 2) Lagrange ----------
def lagrange(x, y):
    x = np.array(x, float); y = np.array(y, float)
    def P(t):
        t = np.asarray(t, float)
        total = np.zeros_like(t)
        for i in range(len(x)):
            Li = np.ones_like(t)
            for j in range(len(x)):
                if j != i:
                    Li *= (t - x[j]) / (x[i] - x[j])
            total += y[i] * Li
        return total
    return P

# ---------- 3) Spline cúbico sujeto (programado a mano) ----------
def tridiagonal(inf, diag, sup, r):
    """Algoritmo de Thomas para un sistema tridiagonal, O(n)."""
    n = len(diag)
    u = np.zeros(n); g = np.zeros(n); z = np.zeros(n)
    u[0] = diag[0]; g[0] = sup[0] / u[0]; z[0] = r[0] / u[0]
    for i in range(1, n):
        u[i] = diag[i] - inf[i - 1] * g[i - 1]
        if i < n - 1:
            g[i] = sup[i] / u[i]
        z[i] = (r[i] - inf[i - 1] * z[i - 1]) / u[i]
    c = np.zeros(n)
    c[-1] = z[-1]
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - g[i] * c[i + 1]
    return c

def spline_sujeto(x, y, d0, dn):
    x = np.array(x, float); a = np.array(y, float)
    n = len(x) - 1
    h = np.diff(x)
    diag = np.zeros(n + 1); inf = np.zeros(n); sup = np.zeros(n); r = np.zeros(n + 1)
    diag[0] = 2 * h[0]; sup[0] = h[0]
    r[0] = 3 * (a[1] - a[0]) / h[0] - 3 * d0
    for i in range(1, n):
        inf[i - 1] = h[i - 1]
        diag[i] = 2 * (h[i - 1] + h[i])
        sup[i] = h[i]
        r[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]
    inf[n - 1] = h[n - 1]; diag[n] = 2 * h[n - 1]
    r[n] = 3 * dn - 3 * (a[n] - a[n - 1]) / h[n - 1]
    c = tridiagonal(inf, diag, sup, r)
    b = (a[1:] - a[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    def S(t):
        t = np.asarray(t, float)
        i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 1)
        dx = t - x[i]
        return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return S

# ---------- Formato de números para LaTeX ----------
def sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10**e
    return "$%.2f\\times10^{%d}$" % (m, e)

def num(v):
    return "$%.2f$" % v

# ---------- Cálculos ----------
res = {}
for nombre, c in curvas.items():
    x, y = np.array(c["x"], float), np.array(c["y"], float)
    t = np.linspace(x[0], x[-1], NMALLA)
    PV, coef, condV = vandermonde(x, y)
    PL = lagrange(x, y)
    S = spline_sujeto(x, y, c["d0"], c["dn"])
    Sref = CubicSpline(x, y, bc_type=((1, c["d0"]), (1, c["dn"])))
    f = {"Vandermonde": PV, "Lagrange": PL, "Spline": S}
    res[nombre] = dict(
        x=x, y=y, t=t, f=f, n=len(x) - 1, condV=condV,
        dVL=np.max(np.abs(PV(t) - PL(t))),
        dSci=np.max(np.abs(S(t) - Sref(t))),
        errnodos={k: np.max(np.abs(g(x) - y)) for k, g in f.items()},
        minimo={k: g(t).min() for k, g in f.items()},
        maximo={k: g(t).max() for k, g in f.items()},
    )
    r = res[nombre]
    print(nombre, f"(n = {r['n']})")
    print(f"   cond(V) = {r['condV']:.3e} | max|PV-PL| = {r['dVL']:.3e} | max|S-SciPy| = {r['dSci']:.3e}")
    for k in f:
        print(f"   {k:12s} err nodos = {r['errnodos'][k]:.2e}, min = {r['minimo'][k]:.2f}, max = {r['maximo'][k]:.2f}")
    print(f"   rango de los datos: [{y.min():.2f}, {y.max():.2f}]")

# ---------- Figuras por curva (3 métodos lado a lado) ----------
for k, (nombre, r) in enumerate(res.items(), start=1):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    for ax, (met, g) in zip(axs, r["f"].items()):
        ax.plot(r["t"], g(r["t"]), "b-")
        ax.plot(r["x"], r["y"], "ro", ms=4)
        ax.set_ylim(*YLIM); ax.set_xlabel("x"); ax.grid(alpha=.3)
        ax.set_title(f"{nombre}: {met}")
    axs[0].set_ylabel("f(x)")
    fig.tight_layout()
    fig.savefig(f"figuras/curva{k}.pdf")
    plt.close(fig)

# ---------- Siluetas completas (3 curvas unidas) por método ----------
for met, archivo in [("Vandermonde", "silueta_vandermonde"),
                     ("Lagrange", "silueta_lagrange"),
                     ("Spline", "silueta_spline")]:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for nombre, r in res.items():
        ax.plot(r["t"], r["f"][met](r["t"]), "b-")
        ax.plot(r["x"], r["y"], "ro", ms=3)
    ax.set_xlim(0, 31); ax.set_ylim(*YLIM)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.grid(alpha=.3)
    ax.set_title(f"Silueta de Snoopy: {met}")
    fig.tight_layout()
    fig.savefig(f"figuras/{archivo}.pdf")
    plt.close(fig)

# ---------- Tablas (solo el tabular; caption y label van en el .tex principal) ----------
for k, (nombre, r) in enumerate(res.items(), start=1):
    ms = ["Vandermonde", "Lagrange", "Spline"]
    L = []
    L.append("\\begin{tabular}{lccc}")
    L.append("  \\toprule")
    L.append("  \\rowcolor{Azul1!50}")
    L.append("   & Vandermonde & Lagrange & Spline \\\\")
    L.append("  \\midrule")
    L.append("  Error en los nodos & " + " & ".join(sci(r["errnodos"][m]) for m in ms) + " \\\\")
    L.append("  Mínimo en la malla & " + " & ".join(num(r["minimo"][m]) for m in ms) + " \\\\")
    L.append("  Máximo en la malla & " + " & ".join(num(r["maximo"][m]) for m in ms) + " \\\\")
    L.append("  \\midrule")
    L.append("  \\multicolumn{4}{l}{Rango de los datos: $[%.2f,\\,%.2f]$} \\\\" % (r["y"].min(), r["y"].max()))
    L.append("  \\bottomrule")
    L.append("\\end{tabular}")
    open(f"tablas/metricas_curva{k}.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

L = []
L.append("\\begin{tabular}{lcccc}")
L.append("  \\toprule")
L.append("  \\rowcolor{Azul1!50}")
L.append("  Curva & $n$ & $\\cond(V)$ & $\\max|P_V-P_L|$ & $\\max|S-S_{\\text{SciPy}}|$ \\\\")
L.append("  \\midrule")
for k, (nombre, r) in enumerate(res.items(), start=1):
    L.append("  %d & %d & %s & %s & %s \\\\" % (k, r["n"], sci(r["condV"]), sci(r["dVL"]), sci(r["dSci"])))
L.append("  \\bottomrule")
L.append("\\end{tabular}")
open("tablas/resumen.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("Listo: figuras/ y tablas/ generadas.")
