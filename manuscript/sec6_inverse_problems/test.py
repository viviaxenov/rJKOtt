import numpy as np
import rJKOtt

import matplotlib.pyplot as plt

dim = 10
grid = rJKOtt.Grid(-3.0, 3.0, 300, dim=dim)

distr = rJKOtt.TensorTrainDistribution.gaussian(grid)


N_test = 1000
x = np.random.randn(N_test, dim)
x = grid.clip_sample(x)

rho, grad_rho = rJKOtt.DistributionOnGrid.tt_on_grid_value_and_gradient(
    x, distr.rho_tt, grid
)

dens_fn = lambda _x: (2.0 * np.pi) ** (-_x.shape[-1] / 2.0) * np.exp(
    -np.linalg.norm(_x, axis=-1) ** 2 / 2.0
)
grad_dens_fn = lambda _x: -dens_fn(_x)[:, np.newaxis] * _x
score_fn = lambda _x: -_x

err_dens = np.abs(dens_fn(x) - rho) / dens_fn(x)
err_grad = np.linalg.norm(grad_dens_fn(x) - grad_rho, axis=-1) / np.linalg.norm(
    grad_dens_fn(x), axis=-1
)
err_score = np.linalg.norm(score_fn(x) - distr.score(x), axis=-1) / np.linalg.norm(
    score_fn(x), axis=-1
)

print(err_dens.shape, err_score.shape, err_grad.shape)

plt.hist(err_dens, alpha=0.5, label="density")
plt.hist(err_grad, alpha=0.5, label="grad")
plt.hist(err_score, alpha=0.5, label="score")
plt.yscale("log")
plt.legend()

fig, axs = distr.plot_matrix_marginals(
    sym=True,
    contour_kwargs={"cmap": "Oranges", "alpha": 0.5},
    plot_kwargs={"color": "tab:orange"},
)


axs[1, 0].quiver(*x[:, :2].T, *grad_rho[:, :2].T, color="tab:orange")
axs[1, 0].quiver(*x[:, :2].T, *grad_dens_fn(x)[:, :2].T, color="tab:green")

plt.show()

print(np.count_nonzero(err_dens < 5e-2) / err_dens.shape[0])
