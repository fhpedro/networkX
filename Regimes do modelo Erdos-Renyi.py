import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Parâmetros
n = 500
num_samples = 80
#k_values = np.linspace(0, 6, 50)
k_values = np.linspace(1e-3, 6, 100)  # começa em 0.001

gcc_means = []
gcc_stds = []  # desvio padrão

# --- Simulação ---
for k in k_values:
    p = k / (n - 1)

    print(f"Calculando para k={k:.3f}, p={p:.5f}")

    gcc_sizes = []
    for _ in range(num_samples):
        G = nx.erdos_renyi_graph(n, p)
        if len(G) == 0:
            gcc_sizes.append(0)
            continue
        largest_cc = max(nx.connected_components(G), key=len)
        gcc_sizes.append(len(largest_cc) / n)
    gcc_means.append(np.mean(gcc_sizes))
    gcc_stds.append(np.std(gcc_sizes))

# --- Teoria ---
def giant_component_fraction(k):
    # resolver S = 1 - exp(-k*S)
    if k <= 1:
        return 0.0
    sol = fsolve(lambda S: S - (1 - np.exp(-k*S)), 0.5)[0]
    return sol

theory_values = [giant_component_fraction(k) for k in k_values]

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(9,6))

# Simulação com barras de erro
ax1.errorbar(k_values, gcc_means, yerr=gcc_stds, fmt='o', color='purple',
             ecolor='gray', elinewidth=1, capsize=3, label="Simulação ER")

# Teoria
ax1.plot(k_values, theory_values, color='red', linestyle='-', linewidth=2, label="Teoria analítica")

# Ponto crítico
ax1.axvline(x=1, color='orange', linestyle='--', label='ponto crítico (k≈1)')

# Labels
ax1.set_xlabel("Grau médio ⟨k⟩")
ax1.set_ylabel("Tamanho relativo da GCC (N_cc / N)")
ax1.set_title(f"Transição de fase no modelo Erdős–Rényi (n={n})")
ax1.grid(alpha=0.3)
ax1.legend()

# Escala logarítmica no eixo x
ax1.set_xscale("log")

# eixo x secundário: probabilidade p
def k_to_p(x): return x / (n - 1)
def p_to_k(x): return x * (n - 1)

ax2 = ax1.secondary_xaxis('top', functions=(k_to_p, p_to_k))
ax2.set_xlabel("Probabilidade p")

plt.show()