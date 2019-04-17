import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt


def get_diff(data):
    '''
    get the difference of spatial data
    input: (n,3) matrix data
    output: np.array(n,2)
    '''

    # calculate the distance between each pair of points
    dist_vec = pdist(data[:, :2])
    z_vec = pdist(data[:, 2:])**2
    # calculate the difference of the values in each pairwise
    diff = np.stack([dist_vec, z_vec])
    index = diff[0] <= diff[0].max()*1/2
    diff = np.array([dif[index] for dif in diff])

    return diff


def emp_variogram(z_vario, lag_h):
    '''
    calculate empirical variogram
    input: difference of spatial (2, nC2)matrix,  bandwith of bins
    '''
    bin_means, bin_edges, bin_number = \
        stats.binned_statistic(z_vario[0], z_vario[1], statistic='mean', bins=lag_h)
    bin_count, bin_edges, bin_number = \
        stats.binned_statistic(z_vario[0], z_vario[1], statistic='count', bins=lag_h)  # NWLS法のためにカウントをとる
    # bin_edgesに関しては最初のものを省く
    e_vario = np.stack([bin_edges[1:], bin_means[0:]], axis=0)
    e_vario = np.delete(e_vario, np.where(e_vario[1] <= 0), axis=1)
    e_vario = e_vario/2

    return e_vario, bin_count


def liner_model(x, a, b):
    return a + b * x


def gaussian_model(x, a, b, c):
    return a + b * (1 - np.exp(-(x / c)**2))  # range param:c


def exponential_model(x, a, b, c):
    return a + b * (1 - np.exp(-(x / c)))  # range param: c


def spherical_model(x, a, b, c):
    cond = [x < c, x > c]
    func = \
        [lambda x: a + (b / 2) * (3 * (x / c) - (x / c)**3), lambda x: a + b]
    # range param: c
    return np.piecewise(x, cond, func)


def auto_fit(e_vario, fitting_range, selected_model):
    '''
    設定してモデルを経験バリオグラムにフィッティングし、パラメータを推定する
    Input:
    Output:
    '''
    # フィッティングレンジまでで標本バリオグラムを削る
    data = np.delete(e_vario, np.where(e_vario[0] > fitting_range)[0], axis=1)
    if (selected_model == 0):
        param, cov = opt.curve_fit(liner_model, data[0], data[1])
    elif (selected_model == 1):
        param, cov = \
            opt.curve_fit(gaussian_model, data[0], data[1], bounds=(0, fitting_range))
    elif (selected_model == 2):
        param, cov = \
            opt.curve_fit(exponential_model, data[0], data[1], bounds=(0, fitting_range))
    elif (selected_model == 3):
        param, cov = \
            opt.curve_fit(spherical_model, data[0], data[1], bounds=(0, fitting_range))
    param = np.insert(param, 0, [selected_model, fitting_range])
    return param


def plot_semivario(e_vario, param):
    fig, ax = plt.subplots()
    ax.plot(e_vario[0], e_vario[1], 'o')
    xlim_arr = np.linspace(0, np.max(e_vario[0])*1.1, 10)
    print(xlim_arr)
    if (param[0] == 0):
        ax.plot(xlim_arr, liner_model(xlim_arr, param[2], param[3]), 'r-')
        print(param[2], param[3])
    elif (param[0] == 1):
        ax.plot(xlim_arr, gaussian_model(xlim_arr, param[2], param[3], param[4]), 'r-')
        print(xlim_arr, param[3], param[4])
    elif (param[0] == 2):
        ax.plot(xlim_arr, exponential_model(xlim_arr, param[2], param[3], param[4]), 'r-')
        print(param[2], param[3], param[4])
    elif (param[0] == 3):
        ax.plot(xlim_arr, spherical_model(xlim_arr, param[2], param[3], param[4]), 'r-')
        print(param[2], param[3], param[4])
    # グラフのタイトルの設定
    ax.set_title('Semivariogram')
    # 軸ラベルの設定
    # ax.set_xlim([0, np.max(e_vario[0])])
    # ax.set_ylim([0, np.max(e_vario[1])])
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Semivariance')
    # グラフの描画
    return fig


def choose_model(e_vario, count, plot=True):
    '''
    NWLS法による理論バリオグラムの推定
    input: empirical variogram, number of data in each bin
    output: param(model, coeff), minimized squared residuals, plot of result
    注意: この関数は不要かもしれない。auto_varioで関数型を指定する?
    '''
    obj_min = None
    model_param = None
    for i in range(0, 4):
        param = auto_fit(e_vario, 100, i)
        #  誤差を算出する
        if i == 0:
            theoritical_vario = liner_model(e_vario[0], param[2], param[3])
            resid = e_vario[1] - theoritical_vario
            resid_sq = resid**2
            weight = count/(theoritical_vario**2)
            # 最小化する
            obj = weight*resid_sq
            obj_sum = obj.sum()
        if i == 1:
            theoritical_vario = gaussian_model(e_vario[0], param[2], param[3], param[4])
            resid = e_vario[1] - theoritical_vario
            resid_sq = resid**2
            weight = count/(theoritical_vario**2)
            # 最小化する
            obj = weight*resid_sq
            obj_sum = obj.sum()
        if i == 2:
            theoritical_vario = exponential_model(e_vario[0], param[2], param[3], param[4])
            resid = e_vario[1] - theoritical_vario
            resid_sq = resid**2
            weight = count/(theoritical_vario**2)
            # 最小化する
            obj = weight*resid_sq
            obj_sum = obj.sum()
        if i == 3:
            theoritical_vario = spherical_model(e_vario[0], param[2], param[3], param[4])
            resid = e_vario[1] - theoritical_vario
            resid_sq = resid**2
            weight = count/(theoritical_vario**2)
            # 最小化する
            obj = weight*resid_sq
            obj_sum = obj.sum()
        if obj_min is None or obj_sum < obj_min:
            obj_min = obj_sum
            model_param = param
    if plot is True:
        best_vario_fig = plot_semivario(e_vario, model_param)
    else:
        best_vario_fig = None
    return model_param, obj_min, best_vario_fig


def auto_vario(data, lag, plot=True):
    '''
    NWLS法に基づいてバリオグラムを推定する
    Input:
    Output:
    '''
    e_vario, count = emp_variogram(data, lag)
    param, resid, vario_plot = choose_model(e_vario, count, plot)
    return param, lag, vario_plot
