import numpy as np
import scipy.spatial.distance as dist
import statsmodels.api as sm


def calc_c0(param, distance, *range_of_linear):
    '''
    有効レンジを使った分散(ナゲット)の推定
    '''

    ef_range = put_effective_range(param, range_of_linear)
    c0 = call_model(ef_range, param)

    return ef_range, c0


def put_effective_range(param, *range_of_linear):
    '''
    有効レンジを計算する. 線形モデルを推定する場合は適当な数値を当てはめる
    input: パラメータ
    output: 有効レンジ
    '''

    if param[0] == 0:
        # linear model
        ef_range = range_of_linear
    elif param[0] == 1:
        # gaussian model
        ef_range = np.sqrt(3)*param[3]
    elif param[0] == 2:
        # exponential model
        ef_range = 3*param[3]
    elif param[0] == 3:
        # spherical model
        ef_range = param[3]

    return ef_range


def call_model(x, param):
    '''
    コバリオグラム関数の呼び出し
    '''

    if param[0] == 0:  # 線形モデル
        func = lambda x: param[2] - param[3]*x
    elif param[0] == 1:  # ガウス型
        func = lambda x: param[2]*np.exp(-(x/param[3])**2)
    elif param[0] == 2:  # 指数型
        func = lambda x: param[2]*np.exp(- (x/param[3]))
    elif param[0] == 3:  # 球形型
        func = lambda x: param[2]*(1 - 3*x/2*param[3] + ((x/param[3])**3)/2)

    return func(x)


def est_covariance_matrix(distance_matrix, param, c0, reg=True):
    '''
    分散共分散行列を推定する
    input: 距離行列, パラメータ, 分散(対角要素)
    output: 分散共分散行列の推定値
    '''
    est_covar = np.vectorize(est_covario, excluded=[1])
    # 距離行列にコバリオグラム関数を適用
    covariance = est_covar(distance_matrix, param, c0)
    # 推定値が正則ではない場合に、正則な行列に変換する
    if reg is True:
        modified_covariance = convert_add_lmd(covariance)
    else:
        modified_covariance = covariance

    return modified_covariance


def calc_distance_matrix(data, point_columns):
    '''
    距離行列を計算する
    input: データ, 緯度経度を表すコラムの名前
    output: 距離行列
    '''

    points = data[point_columns].values
    distance = dist.pdist(points)
    distance = dist.squareform(distance)

    return distance


def est_covario(x, param, c0):
    '''
    コバリオグラムの値を推定する
    input: 距離の値, パラメータ, 分散(ナゲット)
    '''
    cond = [x <= 0, x > 0]
    func = [c0, call_model(x, param)]
    return np.piecewise(x, cond, func)


def convert_add_lmd(mat, eps=0.0001):
    '''
    行列を正則化する
    K' = K + lmd I
    上式に従って, 固有値のうち負の値を取るものを非負値になおす
    '''
    mat_positive_definite = np.copy(mat)
    eigen_values = np.linalg.eigvals(mat)
    min_eigen_values = np.min(eigen_values)
    if min_eigen_values < 0:
        lmd = np.real(-min_eigen_values + eps)  # larger than 0
        print(lmd)
        mat_positive_definite += lmd * np.eye(mat.shape[0])
    return mat_positive_definite


def do_gls(y, X, covariance_matrix):
    '''
    一般化最小二乗法(GLS)
    input: データ, 重みとなる共分散行列
    output: GLSの結果と残差
    '''

    gls_mod = sm.GLS(endog=y, exog=X, sigma=covariance_matrix)
    gls_res = gls_mod.fit()
    residual = gls_res.resid

    return gls_res, residual


def put_pred_val(gls_res, new_data):
    '''
    GLSによる予測値
    '''

    pred = gls_res.predict(exog=new_data)

    return pred


def calc_distance_new_data(new_points, points):
    '''
    既存データの各点と予測したいデータの各点との距離行列を求める
    '''

    m_list = []
    for n_point in new_points:
        pair = [dist.pdist(np.vstack([n_point, point])) for point in points]
        pair = np.array(pair)
        m_list.append(pair)
    c_mat = np.hstack(m_list)

    return c_mat


def do_kriging(fitted_val, c_mat, sigma, resid):
    '''
    クリギング(GLSの値を共分散行列で補正)を実行
    input: GLSの予測値, 共分散行列, 残差(GLSの予測誤差)
    output: 空間補間された値
    '''

    krig_weight = np.dot(c_mat.T, np.linalg.inv(sigma))
    kriging_val = fitted_val + np.dot(krig_weight, resid)

    return kriging_val
