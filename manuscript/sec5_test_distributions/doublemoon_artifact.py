from rJKOtt import *
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

import emcee
import teneva
import itertools

from num2tex import num2tex, configure as num2tex_configure
num2tex_configure(exp_format="cdot", display_singleton=True)

import os

import torch
from geomloss import SamplesLoss
_S2_dist_fn = SamplesLoss(blur=0.3)
S2_dist_fn = lambda _s1, _s2: np.float64(_S2_dist_fn(torch.from_numpy(_s1), torch.from_numpy(_s2))
        )
opath = "./outputs/mixture.npz"
# load the reference chain | if not exists, generate
fname = "./mcmc_results/2d_doublemoon.h5"

dim = 2
# beta = 5e-4
# T = 10. / beta

beta = 1e-4
T = 100.0 / beta
L = 3.0  # Choose the grid bounds
N = 100
n_samples = 600  # in each batch of rJKOtt and MCMC samples
grid = Grid(-L, L, N, dim)  # Initialize the grid
tt_init = TensorTrainDistribution.gaussian(grid)  # standart Gaussiam

target = DenseArrayDistribution.get_double_moon(grid, 2.0)
if os.path.exists(fname):
    reader = emcee.backends.HDFBackend(fname, read_only=True)
    mcmc_result = reader
else:
    init = np.random.randn(n_samples, dim)
    backend = emcee.backends.HDFBackend(fname)
    backend.reset(n_samples, dim)

    sampler = emcee.EnsembleSampler(
        n_samples,
        dim,
        target.log_density,
        moves=[(emcee.moves.GaussianMove(cov=2.0), 1.0)],
        backend=backend,
    )
    state = sampler.run_mcmc(init, 50_000, progress=True)
    mcmc_result = sampler

    print("Computing AC times", flush=True)
    ac_times = mcmc_result.get_autocorr_time(quiet=True)
    t_ac = np.max(ac_times)

    print(f'{sampler.acceptance_fraction=}')
    print(f'{t_ac=:.2e}')

sample_ref = mcmc_result.get_chain()[-1]


solver = TensorTrainSolver(
    target.density,  # The function x -> rho_infty(x)
    tt_init,  # Info on the grid and the initial distribution (in TT)
    posterior_cache_size=int(5e6),
)

solver.params.max_rank = 5  # Max TT rank for all the represented variables
solver.params.cross_rel_diff = 1e-6
solver.params.fp_stopping_rtol = 1e-8
solver.params.cross_nfev_with_posterior = int(4e5)
solver.params.zero_threshold = 1e-40


_, rel_errors, _, _ = solver.step(beta, T, save_history=True)
print(
    f"KL error (w. posterior calls): {solver.KLs[-1]:.2e}\nKL estimate (no posterior calls): {solver.KLs_est[-1]:.2e}"
)
n_calls = solver.n_calls
n_cache = solver.n_cache
n_total = n_calls + n_cache
print(
    f"""
    Cache report:
        Total calls:  {n_total:.2e}
        Real calls:   {n_calls:.2e}
        Cached calls: {n_cache:.2e}
        Acceleration {n_total/n_calls:.2f}
        """
)
tt_distr = solver.get_current_distribution()

solver.params.sampling_ode_rtol = 1e-4
solver.params.sampling_ode_atol = 1e-6

solver.params.sampling_sde_fraction = 5e-3
solver.params.sampling_n_euler_maruyama_steps = 20
x0 = np.random.randn(n_samples, dim)
sample_joint = solver.sample(x0)


solver.params.sampling_sde_fraction = 0.0
x0 = np.random.randn(n_samples, dim)
sample_ode = solver.sample(x0)

dist_str_ode = f"$S_2^2(\\rho_{{N, ode}}, \\rho_{{N, ref}}) = {num2tex(S2_dist_fn(sample_ode, sample_ref)):.2g}$"
dist_str_joint = f"$S_2^2(\\rho_{{N, ode+sde}}, \\rho_{{N, ref}}) = {num2tex(S2_dist_fn(sample_joint, sample_ref)):.2g}$"
ij = (0, 1)

footnote_shift = -3.7
fig, axs = plt.subplots(
    1,
    2,
    sharex=True,
    sharey=True,
)
ax = axs[0]
tt_distr.plot_2d_marginal(ij, axs=ax, colors='k', linewidths=.7)
ax.scatter(*sample_ode[:, ij].T, s=0.5, color='r', )
# ax.scatter(*sample_ref[:, ij].T, s=0.5, color='b', )
ax.set_title("ODE only")
ax.text(x=0.5, y=footnote_shift, s=dist_str_ode, ha='center')
ax = axs[1]
tt_distr.plot_2d_marginal(ij, axs=ax, colors='k', linewidths=.7)
ax.scatter(*sample_joint[:, ij].T, s=0.5, color='r', )
# ax.scatter(*sample_ref[:, ij].T, s=0.5, color='b', )
ax.set_title("ODE and SDE")
ax.text(x=0.5, y=footnote_shift, s=dist_str_joint, ha='center')
for ax in axs:
    ax: plt.Axes
    ax.set_aspect('equal')
    ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.99, bottom=0.01)
fig.tight_layout()
fig.savefig("sampling_artefact.pdf")

plt.show()


