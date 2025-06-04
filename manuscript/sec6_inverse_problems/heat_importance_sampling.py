import numpy as np
import rJKOtt
import teneva

from scipy.stats import norm, gaussian_kde
from arviz import hdi
import emcee

from geomloss import SamplesLoss
import torch
import ot

import matplotlib.pyplot as plt
import matplotlib as mpl

import os

from typing import List, Callable

from heat_equation import forward_heat

import argparse
import itertools

from copy import copy

_S2_dist_fn = SamplesLoss(blur=0.3)
S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(torch.from_numpy(_s1), torch.from_numpy(_s2))


n_draws = 10
n_samples = 400

default_params = rJKOtt.TensorTrainSolverParams()

default_params.zero_threshold = 1e-40
default_params.cross_rel_diff = 1e-5
default_params.fp_max_iter = 100
default_params.fp_method = "picard"
default_params.fp_relaxation = 1.0
default_params.cross_nfev_with_posterior = 2e5
default_params.max_rank = 25
default_params.fp_stopping_rtol = 3e-4

default_params.sampling_ode_rtol = 1e-2
default_params.sampling_ode_atol = 1e-4
default_params.sampling_sde_fraction = 1e-3
default_params.sampling_n_euler_maruyama_steps = 100


def get_posterior(
    forward: Callable,
    xs: np.ndarray,
    ts: np.ndarray,
    theta_true: np.ndarray,
    sigma_measurement=0.1,
    sigma_prior=1.0,
    random_seed=1,
    k=1.0,
    sorting=False,
):
    assert np.all(sigma_prior > 0.0)
    assert np.all(sigma_measurement > 0.0)

    sorting_val = 1e30 if sorting else 0.0

    dim = theta_true.shape[0]
    u_true = forward(ts, xs, theta_true)
    noise = norm(scale=sigma_measurement).rvs(u_true.shape, random_state=random_seed)

    u_meas = u_true + noise

    def potential(theta: np.ndarray):
        u = forward(ts, xs, theta, k=k)

        sorting_barrier = np.where(
            theta[..., :-1] <= theta[..., 1:],
            0.0,
            sorting_val,
        ).sum(axis=-1)

        return (
            0.5 * (((u - u_meas) / sigma_measurement) ** 2).sum(axis=(1, 2))
            + 0.5 * ((theta / sigma_prior) ** 2).sum(axis=(-1,))
            + sorting_barrier
        )

    log_prob = lambda theta: -potential(theta)
    prob = lambda theta: np.exp(-potential(theta))

    posterior_data = {
        "forward": forward,
        "prob_fn": prob,
        "log_prob_fn": log_prob,
        "theta_true": theta_true,
        "u_meas": u_meas,
        "u_true": u_true,
        "xs": xs,
        "ts": ts,
        "k": k,
    }

    return posterior_data


def plot_solution_samples(
    posterior_data: dict,
    theta_groups: List[np.ndarray],
    group_labels: List[str],
    group_plot_args: List,
    t_plot=None,
    x_plot=None,
    n_plot=200,
    suptitle=True,
):
    forward = posterior_data["forward"]
    fig, axs = plt.subplots(1, 1)

    xs = posterior_data["xs"]
    ts = posterior_data["ts"]

    if x_plot is None:
        x_plot = np.linspace(xs.min(), xs.max(), n_plot)
    if t_plot is None:
        t_plot = ts[-1]

    if t_plot in ts:
        # if the time slice is at measurement time, plot measurements
        t_idx = np.argmax(ts == t_plot)
        u_meas = posterior_data["u_meas"][0, t_idx, :]
        axs.plot(xs, u_meas, "r_")

    for thetas, label, plot_args in zip(theta_groups, group_labels, group_plot_args):
        u_sample = forward(t_plot, x_plot, thetas, k=posterior_data["k"])[
            :, 0, :
        ]  # since t is scalar in this fn
        u_mean = u_sample.mean(axis=0)
        u_std = u_sample.std(axis=0)
        args, kwargs = plot_args
        (line,) = axs.plot(x_plot, u_mean, *args, **kwargs)

        if u_sample.shape[0] > 1:
            # axs.fill_between(x_plot, u_mean + 3*u_std, y2=u_mean - 3*u_std, alpha=0.1, color=line.get_color())
            u_min, u_max = (hdi(u_sample, hdi_prob=0.89)).T
            axs.fill_between(x_plot, u_min, y2=u_max, alpha=0.1, color=line.get_color())

        line.set_label(label)

    if suptitle:
        fig.suptitle(f"Solution profile at $t = {t_plot:.2f}$")

    axs.legend()

    return fig, axs


if __name__ == "__main__":
    dir_name_mcmc = "./mcmc_results"

    dim = 10
    sigma_prior = np.array([1.0 / (i + 1) ** 1 for i in range(dim)])
    sigma_measurement = 0.3

    theta_true = norm().rvs((dim,), random_state=1)
    theta_true[1:] *= -1
    theta_true *= sigma_prior
    xs_meas = np.concatenate([np.linspace(-0.9, -0.5, 5), np.linspace(0.5, 0.9, 5)])
    ts_meas = np.linspace(0.0, 0.1, 11, endpoint=True)[1:]

    forward = forward_heat
    cov_mcmc = 0.005

    pname = f"d{dim:02d}_heat_importance_sampling"
    outputs_dir = os.path.join("./outputs", f"{pname}")
    os.makedirs(outputs_dir, exist_ok=True)
    npz_path = os.path.join("./outputs", f"{pname}.npz")
    fname_sample = os.path.join(dir_name_mcmc, f"{pname}_sample.h5")
    fname_importance_sample = os.path.join(
        dir_name_mcmc, f"{pname}_importance_sample.h5"
    )
    posterior_data = get_posterior(
        forward,
        xs_meas,
        ts_meas,
        theta_true,
        sigma_measurement,
        sigma_prior,
    )

    if os.path.exists(fname_sample):
        reader = emcee.backends.HDFBackend(fname_sample, read_only=True)
        mcmc_result = reader
    else:
        os.makedirs(dir_name_mcmc, exist_ok=True)
        init = np.random.randn(n_samples, dim) * sigma_prior[np.newaxis, :]
        backend = emcee.backends.HDFBackend(fname_sample)
        backend.reset(n_samples, dim)

        sampler = emcee.EnsembleSampler(
            n_samples,
            dim,
            posterior_data["log_prob_fn"],
            moves=[(emcee.moves.GaussianMove(cov=cov_mcmc), 1.0)],
            backend=backend,
        )
        state = sampler.run_mcmc(init, 100_000, progress=True)
        mcmc_result = sampler
        with np.printoptions(precision=2):
            print(
                f"Acceptance fraction:    {sampler.acceptance_fraction.min(), sampler.acceptance_fraction.max()}"
            )
            print(f"Autocorrelation times:  {sampler.get_autocorr_time(quiet=True)}")

    print("Computing AC times", flush=True)
    ac_times = mcmc_result.get_autocorr_time(quiet=False)
    t_ac = np.max(ac_times)
    n_thin = int(t_ac)
    print(ac_times)
    print(t_ac)

    samples_mcmc = mcmc_result.get_chain()
    sample_mcmc = samples_mcmc[-1]

    # > Approximate the importance distribution
    def F(theta: np.ndarray) -> np.ndarray:
        # return forward(0.0, 0.0, theta).reshape((-1,))
        return np.sum(theta, axis=-1)

    def importance_true_log_density(theta: np.ndarray) -> np.ndarray:
        return posterior_data["log_prob_fn"](theta) + np.log(np.abs(F(theta)))

    def importance_true_density(theta: np.ndarray) -> np.ndarray:
        return np.exp(importance_true_log_density(theta))

    def importance_opt_weight(theta: np.ndarray) -> np.ndarray:
        return 1.0 / np.abs(F(theta))

    def importance_tt_density(theta: np.ndarray) -> np.ndarray:
        return np.maximum(1e-40, tt_posterior.density(theta)) * np.abs(F(theta))

    def lpr_importance_mcmc(theta):
        fun = F(theta)
        iw = importance_opt_weight(theta)
        return lpr, fun, iw

    blobs_dtype = [("F", np.float64), ("w", np.float64)]

    if os.path.exists(fname_importance_sample):
        reader = emcee.backends.HDFBackend(fname_importance_sample, read_only=True)
        mcmc_result = reader
    else:
        os.makedirs(dir_name_mcmc, exist_ok=True)
        init = sample_mcmc
        backend = emcee.backends.HDFBackend(fname_importance_sample)
        backend.reset(n_samples, dim)

        sampler = emcee.EnsembleSampler(
            n_samples,
            dim,
            lpr_importance_mcmc,
            blobs_dtype=blobs_dtype,
            moves=[(emcee.moves.GaussianMove(cov=cov_mcmc), 1.0)],
            backend=backend,
        )
        state = sampler.run_mcmc(init, 100_000, progress=True)
        mcmc_result = sampler

    ac_times = mcmc_result.get_autocorr_time(quiet=False)
    t_ac = np.max(ac_times)
    n_thin = int(t_ac) * 10
    samples_ref_importance = mcmc_result.get_chain(thin=n_thin, discard=3_000)
    importance_sample_mcmc = samples_ref_importance[-1]

    posterior_fn = posterior_data["prob_fn"]

    n_nodes = 100
    grid = rJKOtt.Grid(-3.5 * sigma_prior, 3.5 * sigma_prior, n_nodes, dim=dim)
    tt_init = rJKOtt.TensorTrainDistribution.gaussian(grid, sigmas=sigma_prior)

    solver_to_smoothed = rJKOtt.TensorTrainSolver(
        posterior_fn,
        tt_init,
        posterior_cache_size=int(1e7),
        solver_params=copy(default_params),
    )
    solver_to_smoothed.params.max_rank = 1
    solver_to_smoothed.params.cross_nfev_with_posterior = 5e4
    solver_to_smoothed.params.fp_stopping_rtol = 3e-3

    beta = 3.0
    T = 1.0 / beta
    solver_to_smoothed.step(beta, T)

    tt_smoothed = solver_to_smoothed.get_current_distribution()
    tt_smoothed = tt_smoothed.adapt_grid()

    left, right, _ = tt_smoothed.grid
    new_means = np.array([(r + l) / 2.0 for r, l in zip(right, left)])
    new_sigmas = np.array([(r - l) / 6.0 for r, l in zip(right, left)])
    tt_new_init = rJKOtt.TensorTrainDistribution.gaussian(
        tt_smoothed.grid, sigmas=new_sigmas, ms=new_means
    )

    solver_to_posterior = rJKOtt.TensorTrainSolver(
        posterior_fn,
        tt_new_init,
        posterior_cache_size=int(1e7),
        solver_params=copy(default_params),
    )

    beta = 1e-4
    T = 10.0 / beta
    solver_to_posterior.step(beta, T)

    tt_posterior = solver_to_posterior.get_current_distribution()

    solver_importance = rJKOtt.TensorTrainSolver(
        importance_tt_density,
        tt_posterior,
        posterior_cache_size=int(1e7),
        solver_params=copy(default_params),
    )

    beta = 1e-4
    T = 10.0 / beta
    solver_importance.step(beta, T)
    tt_importance = solver_importance.get_current_distribution()

    def importance_weight_tt(theta: np.ndarray) -> np.ndarray:
        return posterior_data["prob_fn"](theta) / tt_importance.density(theta)

    # param_subset = [0, 1, 2]
    param_subset = None
    axs = None
    fig, axs = tt_init.plot_matrix_marginals(
        axs=axs,
        param_subset=param_subset,
        sym=True,
        scatter_args=[0.1],
        contour_kwargs={"cmap": "Blues", "linewidths": 0.7, "alpha": 0.9},
        plot_kwargs={"color": "tab:blue", "label": "Initial"},
    )
    fig, axs = tt_new_init.plot_matrix_marginals(
        axs=axs,
        param_subset=param_subset,
        sym=True,
        scatter_args=[0.1],
        scatter_kwargs={"color": "tab:green"},
        contour_kwargs={"cmap": "Greens", "linewidths": 0.7, "alpha": 0.5},
        # plot_kwargs={"color": "tab:green", "label": "Smoothed $\\rho_\\infty$"},
        plot_kwargs={"color": "tab:green", "label": "New init"},
    )
    fig, axs = tt_posterior.plot_matrix_marginals(
        axs=axs,
        param_subset=param_subset,
        sample=sample_mcmc,
        sym=True,
        scatter_args=[0.1],
        scatter_kwargs={"color": "tab:orange"},
        contour_kwargs={"cmap": "Oranges", "linewidths": 0.7, "alpha": 0.5},
        plot_kwargs={"color": "tab:orange", "label": "$\\rho_\\infty$"},
    )
    axs[-1, -1].legend()

    print(
        f"Sinkhorn(MCMC_posterior, MCMC_importance) =  {S2_dist_fn(sample_mcmc, importance_sample_mcmc):.2e}"
    )

    np_samples_file = np.load(npz_path) if os.path.exists(npz_path) else {}
    if all(
        [
            (x in np_samples_file) and (np_samples_file[x].shape[-1] == dim)
            for x in ["samples_tt", "importance_samples_tt"]
        ]
    ):
        samples_tt = np.load(npz_path)["samples_tt"].reshape(-1, dim)
        importance_samples_tt = np.load(npz_path)["importance_samples_tt"].reshape(
            -1, dim
        )
    else:
        samples_tt = np.zeros((0, dim))
        importance_samples_tt = np.zeros((0, dim))

    while importance_samples_tt.shape[0] < n_samples * n_draws:
        print("Sampling from TT", flush=True)
        init = (
            np.random.randn(450, dim) * new_sigmas[np.newaxis, :]
            + new_means[np.newaxis, :]
        )
        # sample_smoothed = solver_to_smoothed.sample(init)
        sample_tt = solver_to_posterior.sample(init)
        importance_sample_tt = solver_importance.sample(sample_tt)

        samples_tt = np.concatenate((samples_tt, sample_tt), axis=0)
        importance_samples_tt = np.concatenate(
            (importance_samples_tt, importance_sample_tt), axis=0
        )
        print(f"{samples_tt.shape=} {importance_samples_tt.shape=}", flush=True)
        np.savez(
            npz_path,
            dim=dim,
            samples_tt=samples_tt,
            importance_samples_tt=importance_samples_tt,
        )

    samples_tt = samples_tt[: n_samples * n_draws]
    samples_tt = samples_tt.reshape((n_draws, n_samples, dim))
    importance_samples_tt = importance_samples_tt[: n_samples * n_draws]
    importance_samples_tt = importance_samples_tt.reshape((n_draws, n_samples, dim))

    Fp = F(samples_mcmc)
    F_mean_true = Fp.mean()
    F_mean = Fp.mean(axis=-1)
    F_std = Fp.std()

    importance_blobs = mcmc_result.get_blobs(
        thin=n_thin,
        discard=3_000,
    )
    # tt importance sample
    Fi_tt = F(importance_samples_tt)
    # wi_tt = importance_weight_tt(importance_samples_tt.reshape(-1, dim)).reshape(
    #     n_draws, n_samples
    # )
    wi_tt = importance_opt_weight(importance_samples_tt)
    Fi_tt_estimator = Fi_tt * wi_tt / np.mean(wi_tt, axis=-1)[:, np.newaxis]
    Fi_tt_mean = Fi_tt_estimator.mean(axis=-1)

    # mcmc importance sample
    Fi, wi = importance_blobs["F"], importance_blobs["w"]
    Fi_estimator = Fi * wi / np.mean(wi, axis=-1)[:, np.newaxis]
    Fi_mean = Fi_estimator.mean(axis=-1)

    fig, axs = plt.subplots(1, 1)
    axs.hist(F_mean, density=True, alpha=0.7, label="Posterior mean")
    # axs.hist(Fi_mean, density=True, alpha=0.5, label="Importance mean, MCMC")
    axs.hist(Fi_tt_mean, density=True, alpha=0.7, label="Importance mean TT")
    axs.axvline(F_mean_true, color="tab:blue", linestyle="--", label="True mean")
    axs.set_yscale("log")
    axs.legend()
    fig.savefig(os.path.join(outputs_dir, f"importance_estimator.pdf"))
    print(f"{F_mean.std()=:.2e}")
    print(f"{Fi_mean.std()=:.2e}")
    sample_tt = samples_tt[-1]
    axs = None
    fig, axs = tt_posterior.plot_matrix_marginals(
        axs=axs,
        param_subset=param_subset,
        sample=sample_tt,
        sym=True,
        scatter_args=[0.3],
        scatter_kwargs={
            "color": "tab:blue",
            "marker": "*",
            "label": "$\\rho_\\infty$, TT",
        },
        contour_kwargs={"cmap": "Blues", "linewidths": 0.7, "alpha": 0.9},
        plot_kwargs={"color": "tab:blue", "label": "$\\rho_\\infty$"},
    )
    fig, axs = tt_posterior.plot_matrix_marginals(
        param_subset=param_subset,
        axs=axs,
        sample=sample_mcmc,
        sym=True,
        scatter_args=[0.3],
        scatter_kwargs={
            "color": "tab:orange",
            "marker": "^",
            "label": "$\\rho_\\infty$, MCMC",
        },
        contour_kwargs={"cmap": "Oranges", "linewidths": 0.7, "alpha": 0.0},
        plot_kwargs={
            "color": "tab:orange",
        },
    )
    axs[-1, -1].legend()
    axs[-1, -2].legend()

    importance_sample_tt = importance_samples_tt[-1]
    axs = None
    fig, axs = tt_posterior.plot_matrix_marginals(
        axs=axs,
        param_subset=param_subset,
        sample=importance_sample_mcmc,
        sym=True,
        scatter_args=[1.5],
        scatter_kwargs={
            "color": "tab:blue",
            "marker": "*",
            "label": "$\\rho_{F, MCMC}$",
        },
        contour_kwargs={"cmap": "Blues", "linewidths": 0.7, "alpha": 0.0},
        plot_kwargs={
            "color": "tab:blue",
        },
    )
    fig, axs = tt_importance.plot_matrix_marginals(
        param_subset=param_subset,
        axs=axs,
        sample=importance_sample_tt,
        sym=True,
        scatter_args=[1.5],
        scatter_kwargs={
            "color": "tab:orange",
            "marker": "o",
            "label": "$\\rho_{F, TT}$",
        },
        contour_kwargs={"cmap": "Oranges", "linewidths": 0.7, "alpha": 0.5},
        plot_kwargs={"color": "tab:orange", "label": "Importance"},
    )
    axs[-1, -2].legend()
    fig.savefig(os.path.join(outputs_dir, f"importance.pdf"))

    samples_mcmc = samples_mcmc[-n_draws:]

    coupling_ref_ref = (
        (samples_mcmc[i], samples_mcmc[j])
        for i in range(n_draws)
        for j in range(i + 1, n_draws)
    )
    dist_ref_tt = [
        S2_dist_fn(s1, s2) for s1, s2 in itertools.product(samples_mcmc, samples_tt)
    ]
    dist_ref_ref = [S2_dist_fn(s1, s2) for s1, s2 in coupling_ref_ref]

    fig, axs = plt.subplots()
    axs.hist(dist_ref_tt, label="tt", alpha=0.7, density=True)
    axs.hist(dist_ref_ref, label="ref", alpha=0.7, density=True)
    axs.legend()
    fig.savefig(os.path.join(outputs_dir, f"sinkhorn_hist.pdf"))

    double_OT_ref_tt = ot.lp.emd2_1d(dist_ref_tt, dist_ref_ref)

    ref_tt_m = np.mean(dist_ref_tt)
    ref_tt_std = np.std(dist_ref_tt)
    ref_ref_m = np.mean(dist_ref_ref)
    ref_ref_std = np.std(dist_ref_ref)

    rk = max(teneva.props.ranks(tt_posterior.rho_tt))

    opath = os.path.join("./outputs", f"{pname}.npz")
    np.savez(
        opath,
        dim=dim,
        ref_tt_m=ref_tt_m,
        ref_tt_std=ref_tt_std,
        ref_ref_m=ref_ref_m,
        ref_ref_std=ref_ref_std,
        double_OT_ref_tt=double_OT_ref_tt,
        samples_mcmc=samples_mcmc,
        samples_tt=samples_tt.reshape(-1, dim),
        importance_samples_tt=importance_samples_tt.reshape(-1, dim),
        rk=rk,
        dist_ref_tt=dist_ref_tt,
    )

    print(
        f"Sinkhorn(TT_importance, MCMC_importance)  =  {S2_dist_fn(importance_sample_tt, importance_sample_mcmc):.2e}"
    )
    print(
        f"Sinkhorn(TT_posterior, MCMC_posterior)   =  {S2_dist_fn(sample_tt, sample_mcmc):.2e}"
    )


    theta_groups = [
        sample_mcmc,
        sample_tt,
        theta_true,
    ]
    labels = [
        "MCMC sample",
        "TT sample",
        "True parameters",
    ]
    args_plot = [
        ([], {"linestyle": "-", "color": "tab:blue", "alpha": 1.0, "linewidth": 0.3}),
        ([], {"linestyle": "-", "color": "tab:orange", "alpha": 1.0, "linewidth": 0.3}),
        (["g-"], {}),
    ]

    L = 1.
    x_plot = np.linspace(-L, L, 200)
    t_plot = [0.0] + list(posterior_data["ts"])
    print(f"\n{t_plot=}\n", flush=True)
    for t in t_plot:
        fig, axs = plot_solution_samples(
            posterior_data,
            theta_groups,
            labels,
            args_plot,
            t_plot=t,
            x_plot=x_plot,
        )
        fig.savefig(os.path.join(outputs_dir, f"solution_samples_{t=:.02f}.pdf"))
        plt.close(fig)
    plt.show()
