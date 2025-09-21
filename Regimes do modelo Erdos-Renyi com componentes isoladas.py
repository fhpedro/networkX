import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parâmetros
n = 500
num_samples = 80
k_values = np.linspace(1e-3, 12, 100)  # começa em 0.001

gcc_means = []
gcc_stds = []
avg_other_sizes = []
std_other_sizes = []

# --- Simulação ---
for k in k_values:
    p = k / (n - 1)
    print(f"Calculando para k={k:.3f}, p={p:.5f}")

    gcc_sizes = []
    other_cc_sizes = []

    for _ in range(num_samples):
        G = nx.erdos_renyi_graph(n, p)
        if len(G) == 0:
            gcc_sizes.append(0)
            other_cc_sizes.append(0)
            continue

        # Todas as componentes conectadas
        components = list(nx.connected_components(G))

        # Componente gigante
        largest_cc = max(components, key=len)
        gcc_sizes.append(len(largest_cc) / n)

        # Outras componentes
        others = [len(c) for c in components if c != largest_cc]
        if others:
            other_cc_sizes.append(np.mean(others))
        else:
            other_cc_sizes.append(0)

    gcc_means.append(np.mean(gcc_sizes))
    gcc_stds.append(np.std(gcc_sizes))
    avg_other_sizes.append(np.mean(other_cc_sizes))
    std_other_sizes.append(np.std(other_cc_sizes))

# --- Teoria (GCC) ---
def giant_component_fraction(k):
    if k <= 1:
        return 0.0
    sol = fsolve(lambda S: S - (1 - np.exp(-k*S)), 0.5)[0]
    return sol

theory_values = [giant_component_fraction(k) for k in k_values]

# --- Plot ---
fig, ax = plt.subplots(figsize=(9,6))

# GCC
ax.errorbar(k_values, gcc_means, yerr=gcc_stds, fmt='o', color='purple',
            ecolor='gray', capsize=3, label="Componente gigante (simulação)")
ax.plot(k_values, theory_values, color='red', linewidth=2, label="Teoria (GCC)")

# Outras componentes
ax.errorbar(k_values, avg_other_sizes, yerr=std_other_sizes, fmt='s', color='blue',
            ecolor='lightblue', capsize=3, label="Componentes menores (simulação)")

# Ponto crítico
ax.axvline(x=1, color='orange', linestyle='--', label='ponto crítico (k≈1)')

# Labels
ax.set_xlabel("Grau médio ⟨k⟩")
ax.set_ylabel("Tamanho relativo (fração GCC) / Tamanho absoluto (outras)")
ax.set_title(f"Transição de fase no modelo Erdős–Rényi (n={n})")
ax.set_xscale("log")
ax.grid(alpha=0.3)
ax.legend()

plt.show()