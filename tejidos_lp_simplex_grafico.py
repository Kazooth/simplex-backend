
"""
Tejidos LP — Simplex (min y max) + Método gráfico + Conclusión

Incluye:
- Maximización con simplex (SciPy linprog)
- Minimización con simplex (SciPy linprog)
- Método gráfico (matplotlib) y verificación por vértices
- Conclusión automática (producción óptima, ganancia, uso de recursos, holguras)

"""
from __future__ import annotations
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from math import isfinite
from dataclasses import dataclass

# -----------------------------
# Parámetros del problema
# -----------------------------
A_max = 500_000  # g
B_max = 300_000  # g
C_max = 108_000  # g

a_std, b_std, c_std = 125, 150, 72    # Estandar (x)
a_prem, b_prem, c_prem = 200, 100, 27 # Premium (y)

p_std, p_prem = 4000, 5000

# -----------------------------
# Utilidades comunes
# -----------------------------
def Z(x: float, y: float) -> float:
    return p_std*x + p_prem*y

def interseccion(L1, L2):
    (a1, b1, c1) = L1
    (a2, b2, c2) = L2
    A = np.array([[a1, b1], [a2, b2]], dtype=float)
    B = np.array([c1, c2], dtype=float)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None
    sol = np.linalg.solve(A, B)
    return sol[0], sol[1]

def vertices_factibles():
    M = np.array([
        [a_std, a_prem],
        [b_std, b_prem],
        [c_std, c_prem],
    ], dtype=float)
    b = np.array([A_max, B_max, C_max], dtype=float)

    rectas = [
        (a_std, a_prem, A_max),
        (b_std, b_prem, B_max),
        (c_std, c_prem, C_max),
        (1.0, 0.0, 0.0),  # x=0
        (0.0, 1.0, 0.0),  # y=0
    ]
    cand = []
    for L1, L2 in itertools.combinations(rectas, 2):
        pt = interseccion(L1, L2)
        if pt is None:
            continue
        x, y = pt
        if not (isfinite(x) and isfinite(y)): 
            continue
        if x >= -1e-9 and y >= -1e-9 and np.all(M @ np.array([x, y]) <= b + 1e-6):
            cand.append((max(0.0, x), max(0.0, y)))
    cand.append((0.0, 0.0))
    # unique
    out = []
    for p in cand:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in out):
            out.append(p)
    return out

# -----------------------------
# Simplex con SciPy
# -----------------------------
@dataclass
class ResultadoLP:
    x: float
    y: float
    z: float
    status: int
    mensaje: str
    metodo: str

def resolver_simplex_max():
    # linprog minimiza: para max Z, minimizamos -Z
    c = np.array([-p_std, -p_prem], dtype=float)
    A_ub = np.array([
        [a_std, a_prem],
        [b_std, b_prem],
        [c_std, c_prem],
    ], dtype=float)
    b_ub = np.array([A_max, B_max, C_max], dtype=float)
    bounds = [(0, None), (0, None)]  # x>=0, y>=0

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    z = -res.fun if res.success else float("nan")
    x, y = (res.x.tolist() if res.success else [float("nan"), float("nan")])
    return ResultadoLP(x=x, y=y, z=z, status=res.status, mensaje=res.message, metodo="simplex-max (HiGHS)")

def resolver_simplex_min():
    # Minimización directa de Z
    c = np.array([p_std, p_prem], dtype=float)
    A_ub = np.array([
        [a_std, a_prem],
        [b_std, b_prem],
        [c_std, c_prem],
    ], dtype=float)
    b_ub = np.array([A_max, B_max, C_max], dtype=float)
    bounds = [(0, None), (0, None)]

    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    z = res.fun if res.success else float("nan")
    x, y = (res.x.tolist() if res.success else [float("nan"), float("nan")])
    return ResultadoLP(x=x, y=y, z=z, status=res.status, mensaje=res.message, metodo="simplex-min (HiGHS)")

# -----------------------------
# Método gráfico
# -----------------------------
def graficar(vertices, optimo_max):
    def interceptos(a, b, c):
        x0 = c/a if a != 0 else np.nan
        y0 = c/b if b != 0 else np.nan
        return x0, y0

    ints = [interceptos(a_std, a_prem, A_max),
            interceptos(b_std, b_prem, B_max),
            interceptos(c_std, c_prem, C_max)]
    x_max = np.nanmax([i[0] for i in ints if np.isfinite(i[0])])
    y_max = np.nanmax([i[1] for i in ints if np.isfinite(i[1])])
    x_plot_max = float(max(10.0, x_max*1.05))
    y_plot_max = float(max(10.0, y_max*1.05))

    xs = np.linspace(0, x_plot_max, 400)
    ya = (A_max - a_std*xs)/a_prem
    yb = (B_max - b_std*xs)/b_prem
    yc = (C_max - c_std*xs)/c_prem

    plt.figure(figsize=(7,6))
    plt.plot(xs, ya, label="125x + 200y = 500000 (A)")
    plt.plot(xs, yb, label="150x + 100y = 300000 (B)")
    plt.plot(xs, yc, label=" 72x +  27y = 108000 (C)")

    # Región factible por muestreo
    X, Y = np.meshgrid(np.linspace(0, x_plot_max, 220), np.linspace(0, y_plot_max, 220))
    mask = (a_std*X + a_prem*Y <= A_max) & (b_std*X + b_prem*Y <= B_max) & (c_std*X + c_prem*Y <= C_max)
    plt.imshow(mask.astype(int), extent=(0, x_plot_max, 0, y_plot_max),
               origin='lower', alpha=0.15, aspect='auto')

    # Vértices
    vx = [v[0] for v in vertices]
    vy = [v[1] for v in vertices]
    plt.scatter(vx, vy, s=30, label="Vértices factibles")

    # Recta objetivo pasando por el óptimo máx del simplex
    xs2 = xs
    y_obj = (optimo_max.z - p_std*xs2)/p_prem
    plt.plot(xs2, y_obj, linestyle='--', label="Recta objetivo en Z* (máx)")

    plt.scatter([optimo_max.x], [optimo_max.y], s=80, marker='*', label="Óptimo Máx (simplex)")
    plt.xlim(0, x_plot_max); plt.ylim(0, y_plot_max)
    plt.xlabel("x = metros de Estandar"); plt.ylabel("y = metros de Premium")
    plt.title("Método gráfico – Región factible y recta objetivo")
    plt.legend(loc="upper right"); plt.grid(True); plt.tight_layout()
    plt.show()

# -----------------------------
# Conclusión
# -----------------------------
def imprimir_conclusion(optimo_max: ResultadoLP, optimo_min: ResultadoLP):
    A_used = a_std*optimo_max.x + a_prem*optimo_max.y
    B_used = b_std*optimo_max.x + b_prem*optimo_max.y
    C_used = c_std*optimo_max.x + c_prem*optimo_max.y
    holg_A = A_max - A_used
    holg_B = B_max - B_used
    holg_C = C_max - C_used

    print("\nCONCLUSIÓN")
    print("----------")
    print(f"[Máx simplex] Z* = ${optimo_max.z:,.0f} en (x, y) = ({optimo_max.x:.2f}, {optimo_max.y:.2f})")
    print(f"Uso de recursos en el máximo:")
    print(f"- A: {A_used:,.0f}/{A_max:,.0f} g (holgura {holg_A:,.0f} g)")
    print(f"- B: {B_used:,.0f}/{B_max:,.0f} g (holgura {holg_B:,.0f} g)")
    print(f"- C: {C_used:,.0f}/{C_max:,.0f} g (holgura {holg_C:,.0f} g)")
    print(f"\n[Mín simplex] Z_min = ${optimo_min.z:,.0f} en (x, y) = ({optimo_min.x:.2f}, {optimo_min.y:.2f})")
    print("- Como los precios son positivos y la región incluye el origen, el mínimo es (0,0) con Z=0.")
    print("- Recursos activos (se agotan) en el máximo: A y B; C queda con holgura.")

# -----------------------------
# Main
# -----------------------------
def main():
    print("Resolviendo con SIMPLEX (SciPy) y graficando...\n")
    # Simplex máx y mín
    opt_max = resolver_simplex_max()
    opt_min = resolver_simplex_min()

    print(f"Max (simplex): status={opt_max.status} -> {opt_max.mensaje}")
    print(f"  x={opt_max.x:.4f}, y={opt_max.y:.4f}, Z={opt_max.z:,.2f}\n")
    print(f"Min (simplex): status={opt_min.status} -> {opt_min.mensaje}")
    print(f"  x={opt_min.x:.4f}, y={opt_min.y:.4f}, Z={opt_min.z:,.2f}\n")

    # Verificación por vértices y tabla
    verts = vertices_factibles()
    rows = [{"x": x, "y": y, "Z": Z(x, y)} for (x, y) in verts]
    df = pd.DataFrame(rows).sort_values("Z", ascending=False).reset_index(drop=True)
    print("Vértices factibles (ordenados por Z desc):\n", df.to_string(index=False))

    # Gráfico
    graficar(verts, opt_max)

    # Conclusión
    imprimir_conclusion(opt_max, opt_min)

if __name__ == "__main__":
    main()
