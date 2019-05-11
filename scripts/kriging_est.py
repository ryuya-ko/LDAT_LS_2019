import calc_variogram as variogram
import numpy as np
import scipy.spatial.distance as dist
import statsmodels.api as sm


def calc_c0(param, distance, *range_of_linear):
    '''
    calculate the common variance(nugget)
    '''

    ef_range = put_effective_range(param, range_of_linear)
    # effective rangeの設定. アドホックに変更しているので要相談
    # if ef_range < np.max(distance):
    #    ef_range = np.max(distance)
    c0 = call_model(ef_range, param)

    return ef_range, c0


def put_effective_range(param, *range_of_linear):
    '''
    put the value of effective range
    if you estimated the linear model, you should arbitrary choose the range
    output: effective range
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
    call the covariogram model
    '''
    # 注意: linear, exponential, sphericalについては未定義
    if param[0] == 0:
        func = lambda x: param[2] - param[3]*x
    if param[0] == 1:
        func = lambda x: param[2]*np.exp(-(x/param[3])**2)
    if param[0] == 2:
        func = lambda x: param[2]*np.exp(- (x/param[3])) 
    if param[0] == 3:
        func = lambda x: param[2]*(1 - 3*x/2*param[3] + ((x/param[3])**3)/2)
    return func(x)


def est_covariance_matrix(distance_matrix, param, c0, reg=True):
    est_covar = np.vectorize(est_covario, excluded=[1])
    # estimate the covariance
    covariance = est_covar(distance_matrix, param, c0)
    # if estimator was not non-negative positive definite, convert it
    # this process is performed only when the matrix is (n, n)
    if reg is True:
        modified_covariance = convert_add_lmd(covariance)
    else:
        modified_covariance = covariance

    '''
    疑問
    半正定値処理をすると、対角要素が0.95周辺に変換される
    しかもc0の値に関してrobust(2.2~3.5の間)
    処理の内容としては機械学習の手法から引っ張ってきた
    '''

    return modified_covariance


def calc_distance_matrix(data, point_columns):
    '''
    calculate the distance matrix of specified points
    input: dataframe, list of columns name
    output: distance matrix, numpy array
    '''

    points = data[point_columns].values
    distance = dist.pdist(points)
    distance = dist.squareform(distance)

    return distance


def est_covario(x, param, c0):
    '''
    to be written
    '''
    cond = [x <= 0, x > 0]
    func = [c0, call_model(x, param)]
    return np.piecewise(x, cond, func)


def convert_add_lmd(mat, eps=0.0001):
    """
    K' = K + lmd I
    lmd is decided as all eigen values are larger than 0
    """
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
    generalized least squares using the est_covariance
    input: data, estimated covariance matrix
    output: result, coeff param, residual
    '''

    gls_mod = sm.GLS(endog=y, exog=X, sigma=covariance_matrix)
    gls_res = gls_mod.fit()
    residual = gls_res.resid

    return gls_res, residual


def put_pred_val(gls_res, new_data):
    '''
    calculate the predicted value by gls
    input: model, exog of new data
    output: predicted values
    '''

    pred = gls_res.predict(exog=new_data)
    return pred


def calc_distance_new_data(new_points, points):
    '''
    calculate the distance matrix bet new data and existing data
    each points should be numpy array, not DataFrame
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
    perform kriging
    input: fitted val, covariance bet new-data, covariance, resid
    output:kriging interpolated values
    '''

    krig_weight = np.dot(c_mat.T, np.linalg.inv(sigma))
    kriging_val = fitted_val + np.dot(krig_weight, resid)

    return kriging_val
