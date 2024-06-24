# import sys
# sys.path.append('../src/')

from rJKOtt import *
from rJKOtt.utility import tt_independent_gaussians
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import teneva

N_comp = 5
dim = 6
L = 3.0  # Choose the grid bounds
N = [
    100,
] * dim  # Number of nodes in the grid; can be chosen independently in each direction, but we take uniforn
grid = Grid(-L, L, N, dim)  # Initialize the grid
means = uniform.rvs(loc=-L / 2, scale=L / 1, size=(N_comp, dim), random_state=1)
sigma = 0.5
covariance = [
    np.eye(
        dim,
    )
    * sigma**2
] * N_comp

gm = GaussianMixture(
    grid,
    means,
    covariance,
)

tt_init = TensorTrainDistribution(
    grid,
    tt_independent_gaussians(  # Gives the density of standard normal in TT format
        [0.0] * dim, [1.0] * dim, [np.linspace(-L, L, _N, endpoint=True) for _N in N]
    ),
)

solver = TensorTrainSolver(
    gm.density,  # The function x -> rho_infty(x)
    tt_init,  # Info on the grid and the initial distribution (in TT)
    TensorTrainSolverParams(),  # Solver's parameters; for the most of them, defaults are OK
)

solver.params.max_rank = 5  # Max TT rank for all the represented variables
solver.params.cross_validation_rtol = 1e-8
solver.params.fp_stopping_rtol = 1e-8
solver.params.fp_relaxation = 0.9

_, rel_errors, _, _ = solver.step(1e-4, 1e5, save_history=True)


print(
    f"KL error (w. posterior calls): {solver.KLs[-1]:.2e}\nKL estimate (no posterior calls): {solver.KLs_est[-1]:.2e}"
)

plt.plot(rel_errors)
plt.yscale("log")
plt.ylabel(r"$\frac{\|\eta_m - \tilde\eta_m\|_2}{\|\eta_m\|_2}$")
plt.xlabel("Iteration $m$")
plt.grid(True)

x0 = np.random.randn(300, dim)

solver.params.sampling_sde_fraction = 1e-3
solver.params.sampling_n_euler_maruyama_steps = 50
sample_tt = solver.sample(x0)
sample_ref = gm.sample(300)

fig, axs = gm.plot_matrix_marginals(
    sym=True,
    sample=sample_ref,
    scatter_args=[0.2],
    scatter_kwargs={"c": "tab:blue"},
    contour_kwargs={
        "cmap": "Blues",
    },
    plot_kwargs={"color": "tab:blue", "label": r"$\rho_\infty$"},
)


tt_distr = solver.get_current_distribution()
tt_distr.plot_matrix_marginals(
    axs=axs,
    sample=sample_tt,
    scatter_args=[0.2],
    scatter_kwargs={"color": "tab:orange"},
    contour_kwargs={
        "cmap": "Oranges",
        "linestyles": "dashed",
    },
    plot_kwargs={"color": "tab:orange", "label": r"$\rho_{TT}$", "linestyle": "--"},
)
axs[-1, -1].legend(bbox_to_anchor=(1.2, 0.5))

plt.show()
