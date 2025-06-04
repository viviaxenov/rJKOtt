from rJKOtt import *
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

from geomloss import SamplesLoss
import torch
import ot

import emcee
import teneva
import itertools

import os

opath = "./outputs/mixture.npz"

N_comp = 5
dim = 30
# beta = 5e-4
# T = 10. / beta

beta = 1e-4
T = 10.0 / beta
L = 3.0  # Choose the grid bounds
N = 30
n_samples = 400  # in each batch of rJKOtt and MCMC samples
grid = Grid(-L, L, N, dim)  # Initialize the grid
tt_init = TensorTrainDistribution.gaussian(grid)  # standart Gaussiam
means = uniform.rvs(loc=-L / 2, scale=L / 1, size=(N_comp, dim), random_state=1)
sigma = 0.5

target = GaussianMixture(grid, means, sigma)


solver = TensorTrainSolver(
    target.density,  # The function x -> rho_infty(x)
    tt_init,  # Info on the grid and the initial distribution (in TT)
    posterior_cache_size=int(5e6),
)

solver.params.max_rank = 5  # Max TT rank for all the represented variables
solver.params.cross_validation_rtol = 0  # disable validation
solver.params.cross_rel_diff = 1e-5
solver.params.fp_stopping_rtol = 1e-6
solver.params.cross_nfev_with_posterior = int(4e5)


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

nstep_short = int(np.ceil(solver.n_calls / n_samples))
print(f"{nstep_short=}")
n_draws = 20
samples_ref = [target.sample(n_samples) for _ in range(n_draws)]

x0 = np.random.randn(n_samples, dim)

solver.params.sampling_ode_rtol = 1e-2
solver.params.sampling_ode_atol = 1e-5
solver.params.sampling_sde_fraction = 1e-3
solver.params.sampling_n_euler_maruyama_steps = 20


_S2_dist_fn = SamplesLoss(blur=0.3)
S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(torch.from_numpy(_s1), torch.from_numpy(_s2))

samples_tt = [solver.sample(np.random.randn(n_samples, dim)) for _ in range(n_draws)]

samples_short = []
for seed in range(n_draws):
    sampler = emcee.EnsembleSampler(
        n_samples,
        dim,
        target.log_density,
        moves=[(emcee.moves.GaussianMove(cov=0.1), 1.0)],
    )
    init = np.random.randn(n_samples, dim)
    state = sampler.run_mcmc(init, nstep_short, progress=True)
    samples_short.append(sampler.get_chain()[-1])


coupling_ref_ref = ((samples_ref[i], samples_ref[j]) for i in range(n_draws) for j in range(i + 1, n_draws) )
coupling_ref_short = ((samples_ref[i], samples_short[j]) for i in range(n_draws) for j in range(i, n_draws) )
coupling_ref_tt = ((samples_ref[i], samples_tt[j]) for i in range(n_draws) for j in range(i, n_draws) )
    
dist_ref_short = [S2_dist_fn(s1, s2) for s1, s2 in coupling_ref_short]
dist_ref_tt = [S2_dist_fn(s1, s2) for s1, s2 in coupling_ref_tt]
dist_ref_ref = [S2_dist_fn(s1, s2) for s1, s2 in coupling_ref_ref]

fig, axs = plt.subplots()
axs.hist(dist_ref_tt, label="tt", alpha=0.7)
axs.hist(dist_ref_ref, label="ref", alpha=0.7)
axs.hist(dist_ref_short, label="short", alpha=0.7)
axs.legend()

fig, axs = target.plot_matrix_marginals(
    sym=True,
    sample=samples_ref[-1],
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
    sample=samples_tt[-1],
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

# Wasserstein distance between distributions of distances

double_OT_ref_tt = ot.lp.emd2_1d(dist_ref_tt, dist_ref_ref)
double_OT_ref_short = ot.lp.emd2_1d(dist_ref_short, dist_ref_ref)

ref_tt_m = np.mean(dist_ref_tt)
ref_tt_std = np.std(dist_ref_tt)
ref_ref_m = np.mean(dist_ref_ref)
ref_ref_std = np.std(dist_ref_ref)
ref_short_m = np.mean(dist_ref_short)
ref_short_std = np.std(dist_ref_short)

min_nsamples = min(arr.shape[0] for arr in samples_tt)
samples_tt = [arr[:min_nsamples, :] for arr in samples_tt]

tt_solution = solver.get_current_distribution()
rk = max(teneva.props.ranks(tt_solution.rho_tt))
ij = [19, 2]
grid_plot = tt_solution.grid.get_2d_grid(*ij)
marg_plot = tt_solution.get_marginal_on_grid(ij)

np.savez(
    opath,
    dim=dim,
    n_cache=solver.n_cache,
    n_calls=solver.n_calls,
    n_calls_mcmc=n_samples * nstep_short,
    ref_tt_m=ref_tt_m,
    ref_tt_std=ref_tt_std,
    ref_ref_m=ref_ref_m,
    ref_ref_std=ref_ref_std,
    ref_short_m=ref_short_m,
    ref_short_std=ref_short_std,
    double_OT_ref_tt=double_OT_ref_tt,
    double_OT_ref_short=double_OT_ref_short,
    samples_short=samples_short,
    samples_ref=samples_ref,
    samples_tt=samples_tt,
    ij=ij,
    grid_plot=grid_plot,
    marg_plot=marg_plot,
    rk=rk,
)

print("Sinkhorn distance:")
print(f"\tRef to TT   : {ref_tt_m:.2f}+-{ref_tt_std:.2f}")
print(f"\tRef to ref  : {ref_ref_m:.2f}+-{ref_ref_std:.2f}")
print(f"\tRef to short: {ref_short_m:.2f}+-{ref_short_std:.2f}")
