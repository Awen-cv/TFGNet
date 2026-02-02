import math
import numpy as np
import random
from scipy.ndimage.interpolation import shift
from scipy.stats import multivariate_normal


def sigma_matrix2(sig_x, sig_y, theta):
    D = np.array([[sig_x**2, 0], [0, sig_y**2]])
    U = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(U, np.dot(D, U.T))


def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)),
                    yy.reshape(kernel_size * kernel_size,
                               1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def cdf2(D, grid):
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, D)
    cdf = rv.cdf(grid)
    return cdf


def bivariate_skew_Gaussian(kernel_size, sig_x, sig_y, theta, D, grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    pdf = pdf2(sigma_matrix, grid)
    cdf = cdf2(D, grid)
    kernel = pdf * cdf
    kernel = kernel / np.sum(kernel)
    return kernel


def mass_center_shift(kernel_size, kernel):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    col_sum, row_sum = np.sum(kernel, axis=0), np.sum(kernel, axis=1)
    delta_h = np.dot(row_sum, ax)
    delta_w = np.dot(col_sum, ax)
    return delta_h, delta_w


def bivariate_skew_Gaussian_center(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   D,
                                   grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    kernel = bivariate_skew_Gaussian(kernel_size, sig_x, sig_y, theta, D, grid)
    delta_h, delta_w = mass_center_shift(kernel_size, kernel)
    kernel = shift(kernel, [-delta_h, -delta_w], mode='nearest')
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_anisotropic_Gaussian(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_isotropic_Gaussian(kernel_size, sig, grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig**2, 0], [0, sig**2]])
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   beta,
                                   grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(
        -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau_type1(kernel_size, sig_x, sig_y, theta, beta, grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(
        np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau_type1_iso(kernel_size, sig, beta, grid=None):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig**2, 0], [0, sig**2]])
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(
        np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_skew_Gaussian_center(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          noise_range=None,
                                          strict=False):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    sigma_max = np.max([sigma_x, sigma_y])
    thres = 3 / sigma_max
    D = [[np.random.uniform(-thres, thres),
          np.random.uniform(-thres, thres)],
         [np.random.uniform(-thres, thres),
          np.random.uniform(-thres, thres)]]

    kernel = bivariate_skew_Gaussian_center(kernel_size, sigma_x, sigma_y,
                                            rotation, D)


    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, D
    else:
        return kernel


def random_bivariate_anisotropic_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          noise_range=None,
                                          strict=False):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    kernel = bivariate_anisotropic_Gaussian(kernel_size, sigma_x, sigma_y,
                                            rotation)


    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation
    else:
        return kernel


def random_bivariate_isotropic_Gaussian(kernel_size,
                                        sigma_range,
                                        noise_range=None,
                                        strict=False):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    kernel = bivariate_isotropic_Gaussian(kernel_size, sigma)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma
    else:
        return kernel


def random_bivariate_generalized_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          beta_range,
                                          noise_range=None,
                                          strict=False):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y,
                                            rotation, beta)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, beta
    else:
        return kernel


def random_bivariate_plateau_type1(kernel_size,
                                   sigma_x_range,
                                   sigma_y_range,
                                   rotation_range,
                                   beta_range,
                                   noise_range=None,
                                   strict=False):
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau_type1(kernel_size, sigma_x, sigma_y, rotation,
                                     beta)


    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, beta
    else:
        return kernel


def random_bivariate_plateau_type1_iso(kernel_size,
                                       sigma_range,
                                       beta_range,
                                       noise_range=None,
                                       strict=False):

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])

    kernel = bivariate_plateau_type1_iso(kernel_size, sigma, beta)

    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma, beta
    else:
        return kernel


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=[0.6, 5],
                         sigma_y_range=[0.6, 5],
                         rotation_range=[-math.pi, math.pi],
                         beta_range=[0.5, 8],
                         noise_range=None):
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_isotropic_Gaussian(
            kernel_size, sigma_x_range, noise_range=noise_range)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_anisotropic_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range)
    elif kernel_type == 'skew':
        kernel = random_bivariate_skew_Gaussian_center(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range)
    elif kernel_type == 'generalized':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_range,
            noise_range=noise_range)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau_type1_iso(
            kernel_size, sigma_x_range, beta_range, noise_range=noise_range)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau_type1(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_range,
            noise_range=noise_range)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def show_one_kernel():
    import matplotlib.pyplot as plt
    kernel_size = 21

    # bivariate skew Gaussian
    D = [[0, 0], [0, 0]]
    D = [[3 / 4, 0], [0, 0.5]]
    kernel = bivariate_skew_Gaussian_center(kernel_size, 2, 4, -math.pi / 4, D)
    # bivariate anisotropic Gaussian
    kernel = bivariate_anisotropic_Gaussian(kernel_size, 2, 4, -math.pi / 4)
    # bivariate anisotropic Gaussian
    kernel = bivariate_isotropic_Gaussian(kernel_size, 1)
    # bivariate generalized Gaussian
    kernel = bivariate_generalized_Gaussian(
        kernel_size, 2, 4, -math.pi / 4, beta=4)

    delta_h, delta_w = mass_center_shift(kernel_size, kernel)
    print(delta_h, delta_w)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs.set_axis_off()
    ax = axs[0][0]
    im = ax.matshow(kernel, cmap='jet', origin='upper')
    fig.colorbar(im, ax=ax)

    # image
    ax = axs[0][1]
    kernel_vis = kernel - np.min(kernel)
    kernel_vis = kernel_vis / np.max(kernel_vis) * 255.
    ax.imshow(kernel_vis, interpolation='nearest')

    _, xx, yy = mesh_grid(kernel_size)
    # contour
    ax = axs[1][0]
    CS = ax.contour(xx, yy, kernel, origin='upper')
    ax.clabel(CS, inline=1, fontsize=3)

    # contourf
    ax = axs[1][1]
    kernel = kernel / np.max(kernel)
    p = ax.contourf(
        xx, yy, kernel, origin='upper', levels=np.linspace(-0.05, 1.05, 10))
    fig.colorbar(p)

    plt.show()


def show_plateau_kernel():
    import matplotlib.pyplot as plt
    kernel_size = 21

    kernel = plateau_type1(kernel_size, 2, 4, -math.pi / 8, 2, grid=None)
    kernel_norm = bivariate_isotropic_Gaussian(kernel_size, 5)
    kernel_gau = bivariate_generalized_Gaussian(
        kernel_size, 2, 4, -math.pi / 8, 2, grid=None)
    delta_h, delta_w = mass_center_shift(kernel_size, kernel)
    print(delta_h, delta_w)


    fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs.set_axis_off()
    ax = axs[0][0]
    im = ax.matshow(kernel, cmap='jet', origin='upper')
    fig.colorbar(im, ax=ax)

    # image
    ax = axs[0][1]
    kernel_vis = kernel - np.min(kernel)
    kernel_vis = kernel_vis / np.max(kernel_vis) * 255.
    ax.imshow(kernel_vis, interpolation='nearest')

    _, xx, yy = mesh_grid(kernel_size)
    # contour
    ax = axs[1][0]
    CS = ax.contour(xx, yy, kernel, origin='upper')
    ax.clabel(CS, inline=1, fontsize=3)

    # contourf
    ax = axs[1][1]
    kernel = kernel / np.max(kernel)
    p = ax.contourf(
        xx, yy, kernel, origin='upper', levels=np.linspace(-0.05, 1.05, 10))
    fig.colorbar(p)

    plt.show()
