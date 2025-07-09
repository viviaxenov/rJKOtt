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
from wave_equation import forward_wave
import solver_euler_hllc


import argparse
import itertools

import numpy as np


def resample_with_log_probs(x, lp, num_samples=None):
    """
    Resample rows of x according to exp(lp) probabilities.

    Args:
        x: (N, d) array of samples
        lp: (N,) array of log probabilities
        num_samples: number of samples to draw (default=N, with replacement)

    Returns:
        resampled x array and corresponding indices
    """
    if num_samples is None:
        num_samples = lp.shape[0]

    # Subtract max log prob for numerical stability
    lp_stable = lp - np.max(lp)

    # Compute softmax (exp and normalize)
    probs = np.exp(lp_stable)
    probs /= probs.sum()

    # Sample indices according to probabilities
    indices = np.random.choice(len(x), size=num_samples, p=probs, replace=True)

    return x[indices]


def forward_euler(ts, xs, thetas, **kwargs):
    res = solver_euler_hllc.euler_hllc.forward(ts, xs, thetas.reshape(-1, 5))
    return res


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
    parameter_bounds=None,
    stab_constant=1.0,
):
    assert np.all(sigma_prior > 0.0)
    assert np.all(sigma_measurement > 0.0)

    sorting_val = 1e30 if sorting else 0.0

    dim = theta_true.shape[0]
    u_true = forward(ts, xs, theta_true.reshape(1, -1))
    axs_to_sum = tuple(range(1, len(u_true.shape)))
    noise = norm(scale=sigma_measurement).rvs(u_true.shape, random_state=random_seed)

    u_meas = u_true + noise

    if parameter_bounds is None:

        def potential(theta: np.ndarray):
            u = forward(ts, xs, theta, k=k)

            sorting_barrier = np.where(
                theta[..., :-1] <= theta[..., 1:],
                0.0,
                sorting_val,
            ).sum(axis=-1)

            return (
                0.5 * (((u - u_meas) / sigma_measurement) ** 2).sum(axis=axs_to_sum)
                + 0.5 * ((theta / sigma_prior) ** 2).sum(axis=(-1,))
                + sorting_barrier
            )

    else:

        def potential(theta: np.ndarray):
            theta = theta.reshape(-1, dim)
            index_in = np.all(
                (theta > parameter_bounds[np.newaxis, :, 0])
                & (theta < parameter_bounds[np.newaxis, :, 1]),
                axis=-1,
            )
            u = forward(ts, xs, theta[index_in, :], k=k)
            ll = 0.5 * (((u - u_meas) / sigma_measurement) ** 2).sum(axis=axs_to_sum)
            lp = np.zeros(theta.shape[0])
            lp[index_in] = ll
            lp[~index_in] = 1e40
            return lp

    log_prob = lambda theta: -potential(theta)
    l0 = log_prob(theta_true)
    prob = lambda theta: np.exp(dim * np.log(stab_constant) + l0 - potential(theta))

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

        if len(u_meas.shape) > 1:
            u_meas = u_meas[..., 0]

    for thetas, label, plot_args in zip(theta_groups, group_labels, group_plot_args):
        print(f"{label=}, {thetas.shape=}")
        if thetas is None:
            continue
        u_sample = forward(t_plot, x_plot, thetas, k=posterior_data["k"])[
            :, 0, :
        ]  # since t is scalar in this fn
        if len(u_sample.shape) > 1:
            u_sample = u_sample[..., 0]
        u_mean = u_sample.mean(axis=0)
        u_std = u_sample.std(axis=0)
        args, kwargs = plot_args
        ll = axs.plot(x_plot, u_mean, *args, **kwargs)
        line = ll[0]

        if u_sample.shape[0] > 1:
            # axs.fill_between(x_plot, u_mean + 3*u_std, y2=u_mean - 3*u_std, alpha=0.1, color=line.get_color())
            u_min, u_max = hdi(u_sample, hdi_prob=0.89).T
            print(f"{ u_min.shape= }")
            print(f"{ u_max.shape= }")
            axs.fill_between(x_plot, u_min, y2=u_max, alpha=0.1, color=line.get_color())

        line.set_label(label)

    if suptitle:
        fig.suptitle(f"Solution profile at $t = {t_plot:.2f}$")

    axs.legend()

    return fig, axs


def plot_preview_multi_refinements(
    distributions: List[List], sample: np.ndarray = None
):
    cmap_names = [
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "Purples",
        "Greys",
    ]
    c_names = [
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:gray",
    ]
    axs = None

    for i, (distr, cname, cmap) in enumerate(zip(distributions, c_names, cmap_names)):
        label = "Init" if i == 0 else f"{i - 1} refinements"
        fig, axs = distr.plot_matrix_marginals(
            axs=axs,
            sym=True,
            contour_kwargs={"cmap": cmap, "linewidths": 0.7},
            plot_kwargs={"color": cname, "label": label},
        )
    if sample is not None:
        fig, axs = distr.plot_matrix_marginals(
            axs=axs,
            sample=sample,
            sym=True,
            scatter_args=[0.1],
            scatter_kwargs={"color": "tab:orange", "marker": "*"},
            contour_kwargs={"alpha": 0.0},
            plot_kwargs={"color": "tab:orange", "linestyle": '-.', 'label': 'MCMC density'},
        )

    axs[-1, -1].legend(
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
        borderaxespad=1.0,
    )

    return fig, axs


def KL_to_sample(
    test_distr: rJKOtt.TensorTrainDistribution,
    sample_true: np.ndarray,
    log_density_true: np.ndarray,
):
    rho_at_sample = test_distr.density(sample_true)
    rho_at_sample = np.maximum(rho_at_sample, 1e-40)
    rho_true = np.exp(log_density_true)
    norm = 1.0 / sample_true.shape[0]
    return np.mean(np.log(rho_at_sample) - log_density_true) + np.log(norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="inverse_problem")
    parser.add_argument("problem", choices=["wave", "heat", "euler"])
    parser.add_argument("-s", "--sort", action="store_true")
    parser.add_argument("-p", "--plotonly", action="store_true")
    args = parser.parse_args()
    sorting = args.sort

    dir_name_mcmc = "./mcmc_results"

    mean_prior = 0.0
    parameter_bounds = None
    N_mcmc_steps = 100_000
    N_resamples = 0
    if args.problem == "wave":
        dim = 6
        sigma_prior = 1.0
        sigma_measurement = 0.3

        theta_true = norm().rvs((dim,), random_state=1)

        xs_meas = np.linspace(
            -2.0,
            2.0,
            10,
        )
        ts_meas = np.linspace(0.0, 2.0, 11, endpoint=True)[1:]

        forward = forward_wave
        cov_mcmc = 0.01
        sigma_prior = np.full((dim,), sigma_prior)

    elif args.problem == "heat":
        dim = 10
        sigma_prior = np.array([1.0 / (i + 1) ** 1 for i in range(dim)])
        sigma_measurement = 0.3

        theta_true = norm().rvs((dim,), random_state=1)
        theta_true[1:] *= -1
        theta_true *= sigma_prior
        xs_meas = np.linspace(
            -0.9,
            0.9,
            10,
        )
        ts_meas = np.linspace(0.0, 0.1, 11, endpoint=True)[1:]

        forward = forward_heat
        cov_mcmc = 0.010

    elif args.problem == "euler":
        N_resamples = 100
        N_mcmc_steps = 500
        dim = 5
        theta_true = np.array(
            [
                5.5,  # x_0 breaking poit
                1.0,
                1.0,  # (rho, p)_L
                0.125,
                0.1,  # (rho, p)_R
            ]
        )
        parameter_bounds = np.array(
            [
                [1.0, 9.0],  # x_0 breaking poit
                [0.80, 1.2],
                [0.80, 1.2],
                [0.03, 0.20],
                [0.03, 0.20],
            ]
        )
        mean_prior = (parameter_bounds[:, 1] + parameter_bounds[:, 0]) / 2.0
        sigma_prior = (parameter_bounds[:, 1] - parameter_bounds[:, 0]) / 6.0
        sigma_measurement = 0.06
        xs_meas = np.linspace(1.0, 9.0, 8, endpoint=True)
        ts_meas = np.array([1.7])
        forward = forward_euler
        cov_mcmc = 0.0002

    pname = f"d{dim:02d}_{'sorted_' if sorting else ''}{args.problem}_ip"
    outputs_dir = os.path.join("./outputs", f"{pname}")
    os.makedirs(outputs_dir, exist_ok=True)
    fname_sample = os.path.join(
        dir_name_mcmc,
        f"{pname}_sample.h5",
    )
    posterior_data = get_posterior(
        forward,
        xs_meas,
        ts_meas,
        theta_true,
        sigma_measurement,
        sigma_prior,
        sorting=sorting,
        parameter_bounds=parameter_bounds,
    )
    print(posterior_data["prob_fn"](posterior_data["theta_true"] * 1.0))
    print("posterior ok ", flush=True)

    n_samples = 400

    if os.path.exists(fname_sample):
        reader = emcee.backends.HDFBackend(fname_sample, read_only=True)
        mcmc_result = reader
    else:
        os.makedirs(dir_name_mcmc, exist_ok=True)
        init = (
            np.random.randn(n_samples, dim) * sigma_prior[np.newaxis, :] + mean_prior
            if parameter_bounds is None
            else np.random.uniform(
                size=(n_samples, dim),
                low=parameter_bounds[:, 0],
                high=parameter_bounds[:, 1],
            )
        )
        if sorting:
            init.sort(axis=-1)
        backend = emcee.backends.HDFBackend(fname_sample)
        backend.reset(n_samples, dim)

        print("Starting sampler")
        sampler = emcee.EnsembleSampler(
            n_samples,
            dim,
            posterior_data["log_prob_fn"],
            moves=[(emcee.moves.GaussianMove(cov=cov_mcmc), 1.0)],
            backend=backend,
            vectorize=True,
        )
        print("Initialized")
        for i in range(N_resamples + 1):
            state = sampler.run_mcmc(init, N_mcmc_steps, progress=True)
            x_cur = sampler.get_chain()[-1, :, :]
            lp_cur = sampler.get_log_prob()[-1, :]
            print(f"Resampling #{i + 1}")
            if i + 1 > N_resamples:
                break
            init = resample_with_log_probs(x_cur, lp_cur)

        mcmc_result = sampler
        with np.printoptions(precision=2):
            print(
                f"Acceptance fraction:    {sampler.acceptance_fraction.min(), sampler.acceptance_fraction.max()}"
            )
            print(f"Autocorrelation times:  {sampler.get_autocorr_time(quiet=True)}")

    print("Computing AC times", flush=True)
    ac_times = mcmc_result.get_autocorr_time(quiet=True)
    t_ac = np.max(ac_times)
    n_thin = int(t_ac)
    print(ac_times)
    print(t_ac)

    sample_mcmc = mcmc_result.get_chain()[-1].reshape(-1, dim)
    sample_lp = mcmc_result.get_log_prob()[-1]

    posterior_fn = posterior_data["prob_fn"]

    n_nodes = 100
    if parameter_bounds is None:
        grid = rJKOtt.Grid(
            -3.0 * sigma_prior + mean_prior,
            3.0 * sigma_prior + mean_prior,
            n_nodes,
            dim=dim,
        )
    else:
        grid = rJKOtt.Grid(
            parameter_bounds[:, 0], parameter_bounds[:, 1], n_nodes, dim=dim
        )

    tt_init = rJKOtt.TensorTrainDistribution.gaussian(grid, sigmas=sigma_prior, ms=mean_prior)

    solver = rJKOtt.TensorTrainSolver(
        posterior_fn, tt_init, posterior_cache_size=int(1e7)
    )
    solver.params.zero_threshold = 1e-40
    solver.params.cross_rel_diff = 1e-5
    solver.params.fp_max_iter = 20
    n_calls = 0
    n_cache = 0

    if args.problem == "heat":
        solver.params.cross_nfev_with_posterior = 5e3
        solver.params.max_rank = 1
        solver.params.fp_stopping_rtol = 1e-3
        solver.params.fp_relaxation = 0.95

        beta = 1.0
        T = 1.0 / beta
        solver.step(beta, T)

        solver.params.fp_method = "picard"
        solver.params.fp_relaxation = 1.0

        beta = 1e-2
        T = 10.0 / beta
        solver.step(beta, T)

    elif args.problem == "wave":
        solver.params.cross_nfev_with_posterior = 5e3
        solver.params.max_rank = 1
        # solver.params.fp_method = "picard"
        # solver.params.fp_relaxation = 1.0
        solver.params.fp_stopping_rtol = 1e-8

        beta = 1e-3
        T = 100.0 / beta
        solver.step(beta, T)

    elif args.problem == "euler":
        solver.params.zero_threshold = 1e-40
        solver.params.cross_nfev_with_posterior = 5e3
        solver.params.max_rank = 6
        solver.params.fp_stopping_rtol = 1.0e-2
        solver.params.fp_max_iter = 20

        # beta = 1.0
        beta = 1.0e-3
        T = 10.0 / beta
        solver.step(beta, T)


        # Plot solution 0 refinements
        tt_coarse_grid = solver.get_current_distribution()
        fig, axs = tt_coarse_grid.plot_matrix_marginals(
            sample=sample_mcmc,
            sym=True,
            scatter_args=[0.1],
            scatter_kwargs={"color": "tab:orange", "marker": "*"},
            contour_kwargs={"cmap": "Blues", "linewidths": 0.7},
            plot_kwargs={"color": "tab:blue"},
        )
        fig.savefig(os.path.join(outputs_dir, "preview_0ref.pdf"))

    n_calls += solver.n_calls
    n_cache += solver.n_cache
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

    tt_solution = solver.get_current_distribution()

    ml_index_mcmc = np.argmax(sample_lp)
    ml_index_tt = teneva.optima_tt(tt_solution.rho_tt)[2]
    theta_mle_mcmc = sample_mcmc[ml_index_mcmc]
    theta_mean_mcmc = np.mean(sample_mcmc, axis=0)
    theta_mle_tt = teneva.ind_to_poi(ml_index_tt, *grid)
    # theta_mean_tt = np.mean(sample_tt, axis=0)
    print("Computing credible intervals", flush=True)
    credible_interval_prob = 0.89
    tt_intervals = np.array(
        [
            tt_solution.get_credible_interval(i, credible_interval_prob)
            for i in range(dim)
        ]
    )
    print("Done!", flush=True)
    mcmc_intervals = hdi(sample_mcmc, credible_interval_prob)
    if args.problem == "wave":
        theta_mle_mcmc.sort()
        theta_mle_tt.sort()

    solver.params.sampling_ode_rtol = 1e-2
    solver.params.sampling_ode_atol = 1e-3

    if args.problem == "heat":
        solver.params.sampling_sde_fraction = 1e-3
        solver.params.sampling_n_euler_maruyama_steps = 200
    elif args.problem == "wave":
        solver.params.sampling_sde_fraction = 1e-3
        solver.params.sampling_n_euler_maruyama_steps = 200
    elif args.problem == "euler":
        solver.params.sampling_sde_fraction = 0.0
        # solver.params.sampling_n_euler_maruyama_steps = 10

    n_thin = int(t_ac) * 10
    samples_ref = mcmc_result.get_chain(thin=n_thin, discard=3_000)
    print(f"{samples_ref.shape=}")
    n_indep = samples_ref.shape[0]
    n_draws = 10
    n_draws = min(n_draws, n_indep // 2)
    samples_ref = samples_ref[-n_draws:]

    if args.plotonly:
        n_draws = 1
    samples_tt = np.zeros((0, dim))
    while samples_tt.shape[0] < n_samples * n_draws:
        if args.problem == "euler":
            l, u, _ = tt_solution.grid
            means = (l + u) / 2.0
            sigma = (u - l) / 6.0
        else:
            means = np.zeros((dim))
            sigma = sigma_prior
        init = np.random.randn(100, dim) * sigma[np.newaxis, :] + means[np.newaxis, :]
        # if sorting:
        #     init.sort(axis=-1)
        sample_new = solver.sample(init)
        # density_at_sample = tt_solution.density(sample_new)
        # pos_idx = density_at_sample >= 1e-10
        # sample_new = sample_new[pos_idx, :]
        samples_tt = np.concatenate((samples_tt, solver.sample(init)), axis=0)
        print(samples_tt.shape, flush=True)
        if samples_tt.shape[0] == 0:
            print(f"Couldn't produce TT samples")
            break
    samples_tt = samples_tt[: n_samples * n_draws]
    if samples_tt.shape[0] != 0:
        samples_tt = samples_tt.reshape((n_draws, n_samples, dim))
        sample_tt = samples_tt[-1]
    else:
        sample_tt = None
    print("Plot sample done", flush=True)
    

    fig, axs = tt_solution.plot_matrix_marginals(
        sample=sample_tt,
        sym=True,
        scatter_args=[0.1],
        scatter_kwargs={"color": "tab:blue", "marker": "*"},
        contour_kwargs={"cmap": "Blues", "linewidths": 0.7},
        plot_kwargs={"color": "tab:blue"},
    )

    for i, ((theta_l, theta_r), theta_mle) in enumerate(
        zip(tt_intervals, theta_mle_tt)
    ):
        ax = axs[i, i]
        density_line = ax.get_lines()[0]
        x, y = density_line.get_data()
        color = density_line.get_color()
        idx_interval = (x >= theta_l) & (x <= theta_r)
        xi = x[idx_interval]
        yi = y[idx_interval]
        ax.fill_between(xi, yi, -1, color=color, alpha=0.2)
        ax.axvline(theta_mle, color=color, linestyle="--")
        ax.set_ylim(0.0, ax.get_ylim()[1])
        L_interval = np.abs(theta_r - theta_l)
        L_grid = tt_solution.grid.right[i] - tt_solution.grid.left[i]
        if args.problem == "wave":
            L = 0.3 * L_interval
        elif args.problem == "heat":
            L = 0.1 * L_grid
        elif args.problem == "euler":
            L = 0.3 * L_grid
        xmin = max(theta_l - L, grid.left[i])
        xmax = min(theta_r + L, grid.right[i])

        for j in range(0, dim):
            ax = axs[j, i]
            ax.set_xlim(xmin, xmax)
            if i == j:
                continue
            ax = axs[i, j]
            ax.set_ylim(xmin, xmax)

        fig.subplots_adjust(
            left=0.0, top=1.0, right=1.0, bottom=0.0, wspace=0.1, hspace=0.1
        )

    fig.savefig(os.path.join(outputs_dir, "distribution.pdf"))

    theta_groups = [
        sample_mcmc,
        sample_tt,
        theta_mle_mcmc,
        theta_mle_tt,
        theta_true,
    ]
    labels = [
        "MCMC sample",
        "TT sample",
        r"MCMC $\theta_{MAP}$",
        r"TT   $\theta_{MAP}$",
        "True parameters",
    ]
    args_plot = [
        ([], {"linestyle": "-", "color": "tab:blue", "alpha": 1.0, "linewidth": 0.3}),
        ([], {"linestyle": "-", "color": "tab:orange", "alpha": 1.0, "linewidth": 0.3}),
        ([], {"linestyle": "--", "color": "tab:blue"}),
        ([], {"linestyle": "--", "color": "tab:orange"}),
        (["g-"], {}),
    ]

    L = sigma_prior[0] * 3.0 if args.problem == "wave" else 1.0
    x_plot = np.linspace(-L, L, 200)
    if args.problem == "euler":
        x_plot = np.linspace(0, 10.0, 200)
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

    n_steps_short = int(np.ceil(n_calls / n_samples))

    if args.plotonly:
        exit()

    samples_short = []
    for seed in range(n_draws):
        sampler = emcee.EnsembleSampler(
            n_samples,
            dim,
            posterior_data["log_prob_fn"],
            moves=[(emcee.moves.GaussianMove(cov=cov_mcmc), 1.0)],
        )
        init = np.random.randn(n_samples, dim) * sigma_prior[np.newaxis, :]
        if sorting:
            init.sort(axis=-1)
        state = sampler.run_mcmc(init, n_steps_short, progress=True)
        samples_short.append(sampler.get_chain()[-1])

    _S2_dist_fn = SamplesLoss(blur=0.3)
    S2_dist_fn = lambda _s1, _s2: _S2_dist_fn(
        torch.from_numpy(_s1), torch.from_numpy(_s2)
    )

    coupling_ref_ref = (
        (samples_ref[i], samples_ref[j])
        for i in range(n_draws)
        for j in range(i + 1, n_draws)
    )
    dist_ref_short = [
        S2_dist_fn(s1, s2) for s1, s2 in itertools.product(samples_ref, samples_short)
    ]
    dist_ref_tt = [
        S2_dist_fn(s1, s2) for s1, s2 in itertools.product(samples_ref, samples_tt)
    ]
    dist_ref_ref = [S2_dist_fn(s1, s2) for s1, s2 in coupling_ref_ref]

    fig, axs = plt.subplots()
    axs.hist(dist_ref_tt, label="tt", alpha=0.7, density=True)
    axs.hist(dist_ref_ref, label="ref", alpha=0.7, density=True)
    axs.hist(dist_ref_short, label="short", alpha=0.7, density=True)
    axs.legend()

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

    rk = max(teneva.props.ranks(tt_solution.rho_tt))

    opath = os.path.join("./outputs", f"{pname}.npz")
    np.savez(
        opath,
        dim=dim,
        n_cache=n_cache,
        n_calls=n_calls,
        n_calls_mcmc=n_samples * n_steps_short,
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
        intervals_mcmc=mcmc_intervals,
        intervals_tt=tt_intervals,
        theta_mle_mcmc=theta_mle_mcmc,
        theta_mle_tt=theta_mle_tt,
        theta_true=theta_true,
        rk=rk,
        dist_ref_tt=dist_ref_tt,
    )
    print(f"{solver.KLs[-1]=:.2e}")
    print(f"{solver.KLs_est[-1]=:.2e}")
    KL_sample = KL_to_sample(tt_solution, sample_mcmc, sample_lp)
    print(f"{KL_sample=:.2e}")

    print("Sinkhorn distance:")
    print(f"\tRef to TT   : {ref_tt_m:.2e}+-{ref_tt_std:.2e}")
    print(f"\tRef to ref  : {ref_ref_m:.2e}+-{ref_ref_std:.2e}")
    print(f"\tRef to short: {ref_short_m:.2e}+-{ref_short_std:.2e}")
    print(f"Double OT:")
    print(f"\tRef to TT   : {double_OT_ref_tt:.2e}")
    print(f"\tRef to short: {double_OT_ref_short:.2e}")

    plt.show()
