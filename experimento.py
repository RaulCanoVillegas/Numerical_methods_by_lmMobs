"""
experimento.py -- Corre el experimento de la tarea para las seis matrices.

Metodo por matriz (segun su estructura):
    cauchy, hilb, moler  -> Cholesky    (si falla: LU con pivoteo, marcado como respaldo)
    kahan                -> sustitucion hacia atras
    chebvand, forsythe   -> LU con pivoteo parcial
Metricas (norma 2):
    error   = ||x_calc - 1|| / ||1||
    residuo = ||b - A x_calc|| / ||b||
    cond    = cond_2(A)
    cota    = cond * eps
"""
import os
import numpy as np
import pandas as pd

from matrices import MATRICES, sistema, EPS
from solvers import (resolver_LU, resolver_chol, resolver_tri,
                     NoDefinidaPositiva, MatrizSingular)

TAMANOS = list(range(10, 201, 20))          # n = 10, 30, ..., 190
PISO = 1e-18                                # para graficar ceros en escala log

METODO = {"cauchy": "chol", "hilb": "chol", "moler": "chol",
          "kahan": "tri", "chebvand": "lu", "forsythe": "lu"}
NOMBRE_METODO = {"chol": "Cholesky", "tri": "Sust. hacia atrás", "lu": "LU con pivoteo"}
TITULO = {"cauchy": "Cauchy", "chebvand": "Chebyshev–Vandermonde", "forsythe": "Forsythe",
          "hilb": "Hilbert", "kahan": "Kahan", "moler": "Moler"}


def resolver(nombre, A, b):
    """Devuelve (x, metodo_usado, respaldo). x es None si el metodo no pudo resolver."""
    m = METODO[nombre]
    try:
        if m == "chol":
            try:
                return resolver_chol(A, b), "chol", False
            except NoDefinidaPositiva:
                return resolver_LU(A, b), "lu", True          # caso de respaldo
        if m == "tri":
            return resolver_tri(A, b), "tri", False
        return resolver_LU(A, b), "lu", False
    except MatrizSingular:
        return None, m, False


def correr(nombre, tamanos=TAMANOS):
    filas = []
    for n in tamanos:
        A, b = sistema(MATRICES[nombre](n))
        xe = np.ones(n)
        with np.errstate(all="ignore"):
            x, usado, resp = resolver(nombre, A, b)
            c = np.linalg.cond(A)
            if x is None:
                err = res = np.nan
            else:
                err = np.linalg.norm(x - xe) / np.linalg.norm(xe)
                res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        filas.append(dict(n=n, cond=c, error=err, residuo=res, cota=c * EPS,
                          metodo=usado, respaldo=resp))
    return pd.DataFrame(filas)


def correr_todo(tamanos=TAMANOS):
    return {nombre: correr(nombre, tamanos) for nombre in MATRICES}


# ---------------------------------------------------------------------------
# Salida a LaTeX
# ---------------------------------------------------------------------------
def sci(x):
    """Formato LaTeX: 1.23e+16 -> $1.23\\times10^{16}$."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x == 0:
        return "$0$"
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def tabla_tex(df):
    lineas = [r"\centering", r"\small",
              r"\begin{tabular}{r l c c c}", r"\toprule",
              r"\rowcolor{Azul1!50}",
              r"$n$ & Método & $\cond(A)$ & Error relativo & Residuo relativo \\",
              r"\midrule"]
    for _, f in df.iterrows():
        n = f"{int(f.n)}" + (r"$^{*}$" if f.respaldo else "")
        lineas.append(f"{n} & {NOMBRE_METODO[f.metodo]} & {sci(f.cond)} & "
                      f"{sci(f.error)} & {sci(f.residuo)} \\\\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    if df.respaldo.any():
        lineas += [r"\par\vspace{2pt}",
                   r"{\footnotesize $^{*}$ Cholesky falló; se resolvió con LU con pivoteo parcial.}"]
    return "\n".join(lineas) + "\n"


def tabla_resumen_tex(resultados, ns=(10, 190)):
    lineas = [r"\centering", r"\small",
              r"\begin{tabular}{l r l c c c}", r"\toprule",
              r"\rowcolor{Azul1!50}",
              r"Matriz & $n$ & Método & $\cond(A)$ & Error relativo & Residuo relativo \\",
              r"\midrule"]
    for nombre, df in resultados.items():
        for n in ns:
            f = df[df.n == n].iloc[0]
            nn = f"{n}" + (r"$^{*}$" if f.respaldo else "")
            lineas.append(f"\\texttt{{{nombre}}} & {nn} & {NOMBRE_METODO[f.metodo]} & "
                          f"{sci(f.cond)} & {sci(f.error)} & {sci(f.residuo)} \\\\")
        lineas.append(r"\addlinespace" if nombre != list(resultados)[-1] else "")
    lineas += [r"\bottomrule", r"\end{tabular}",
               r"\par\vspace{2pt}",
               r"{\footnotesize $^{*}$ Cholesky falló; se resolvió con LU con pivoteo parcial.}"]
    return "\n".join(lineas) + "\n"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura(nombre, df, ruta=None, mostrar=False):
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    n = df.n.values
    ax.semilogy(n, df.cond, "-o", color="#143C78", label=r"cond$(A)$", ms=4)
    ax.semilogy(n, df.cota, "--", color="#7a7a7a", label=r"cond$(A)\,\varepsilon_{\mathrm{máq}}$")
    for col, color, etiq in (("error", "#c0392b", "error relativo"),
                             ("residuo", "#2e8b57", "residuo relativo")):
        y = np.where(df[col].values <= PISO, PISO, df[col].values)   # ceros -> piso
        ax.semilogy(n, y, "-s", color=color, label=etiq, ms=4)
        # casos de respaldo: marcador hueco
        r = df.respaldo.values
        if r.any():
            ax.semilogy(n[r], y[r], "s", mfc="white", mec=color, ms=8, mew=1.3)
    if df.respaldo.any():
        ax.plot([], [], "s", mfc="white", mec="k", label="respaldo (LU)")
    ax.set_xlabel("$n$")
    ax.set_title(TITULO[nombre])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    if ruta:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        fig.savefig(ruta)
    if not mostrar:
        plt.close(fig)
    return fig


def guardar_todo(resultados, carpeta_tablas="tablas", carpeta_figuras="figuras"):
    os.makedirs(carpeta_tablas, exist_ok=True)
    os.makedirs(carpeta_figuras, exist_ok=True)
    for nombre, df in resultados.items():
        with open(f"{carpeta_tablas}/{nombre}.tex", "w", encoding="utf-8") as f:
            f.write(tabla_tex(df))
        figura(nombre, df, ruta=f"{carpeta_figuras}/{nombre}.pdf")
    with open(f"{carpeta_tablas}/resumen.tex", "w", encoding="utf-8") as f:
        f.write(tabla_resumen_tex(resultados))


if __name__ == "__main__":
    res = correr_todo()
    guardar_todo(res)
    pd.set_option("display.width", 160)
    for nombre, df in res.items():
        print(f"\n== {nombre} ==")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2e}"))