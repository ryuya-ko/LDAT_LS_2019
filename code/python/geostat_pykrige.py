#%%
import folium
import geopandas
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pykrige as pk

from shapely.geometry import Point

#%%
dat = pd.read_csv('/Users/mfjkou/sengokuLab/study/LDAT_LS_2019/data/dev/house_price_raw.csv')
mdat = pd.read_csv('/Users/mfjkou/sengokuLab/study/LDAT_LS_2019/data/dev/house_price_pred.csv')

#%%
mdat.head()
#%% [markdown]
# ## バリオグラム推定

#%% [markdown]
# ### バリオグラム雲
# - packageに実装なし
# - 別途調査・実装

#%% [markdown]
# ### 経験バリオグラム

#%% [markdown]
# ## Regression kriging
# - pykrige packageのRegression krigingはsklearnのモデルのみ受け取る
# - バリオグラム単体を表示・操作するクラスはなく, krigingクラスにvariogramの操作機能が包括されている


#%%
from sklearn.linear_model import LinearRegression

#%%
model = LinearRegression(fit_intercept=False)

#%%
y = dat['price'].apply(lambda x: np.log(x)).values
prop = dat[['station', 'tokyo']].values
latlon = dat[['px', 'py']].values
mprop = mdat[['station', 'tokyo']].values
mlatlon = mdat[['px', 'py']].values
#%%
def transform_data(data, columns_latlon, columns_indep):
    '''
    Change the data format to use in kriging
    Input: list of columns:latlon and independent variable
    Output: values in np.array
    '''
    latlon = data[columns_latlon].values
    indeps = data[columns_indep].values
    return latlon, indeps
#%%
latlon, prop = transform_data(dat, ['px', 'py'], ['station', 'tokyo'])
mlatlon, mprop = transform_data(mdat, ['px', 'py'], ['station', 'tokyo'])
y = dat['price'].apply(lambda x: np.log(x)).values
#%%
y

#%%
from pykrige.rk import RegressionKriging
gmodel = RegressionKriging(regression_model=model,  variogram_model='exponential', nlags=15)

#%%
gmodel.fit(prop, latlon, y)

#%%
my = gmodel.predict(mprop, mlatlon)
print(my)

#%%
# LOO cross validation
pred_list = []
for i in range(0,len(dat)):
    dsub = dat.drop(i)
    msub = dat.iloc[i:i+1, :]

    dprop = dsub[['station', 'tokyo']].values
    dlatlon = dsub[['px', 'py']].values
    dy = dsub['price'].apply(lambda x: np.log(x)).values

    m_prop = msub[['station', 'tokyo']].values
    m_latlon = msub[['px', 'py']].values
    gmodel.fit(dprop, dlatlon, dy)
    pred = gmodel.predict(m_prop, m_latlon)
    pred_list.append(pred)

#%%
len(pred_list)

#%%
plt.scatter(y, pred_list)
plt.ylim = (11.8, 12.8)
plt.savefig('/Users/mfjkou/sengokuLab/study/LDAT_LS_2019/data/dev/plot/res_pykrige.png')

#%%
mdat['price'] = my

#%%
mdat.head()
#%%
# geopandas型にする
mdat['Coordinates'] = list(zip(mdat.px, mdat.py))
mdat['Coordinates'] = mdat['Coordinates'].apply(Point)
#%%
mdat.head()

#%%
gmdat = geopandas.GeoDataFrame(mdat, geometry='Coordinates')

#%%
gmdat.head()

#%%
gmdat.plot()

#%%
np.linspace(gmdat['price'].min(), gmdat['price'].max(), 10)

#%%
def set_rang(data, column, num):
    max_val = data[column].max()
    min_val = data[column].min()
    rang = np.linspace(min_val, max_val, num=num)
    return rang

#%%
rang = set_rang(gmdat, 'price', 10)

#%%
gmdat_list = []
for i in range(0, len(rang)):
    if i< len(rang)-1:
        g = gmdat[(gmdat['price'] >= rang[i]) & (gmdat['price'] < rang[i+1])]
    else:
        g = gmdat[gmdat['price'] >= rang[i]]
    gmdat_list.append(g)

#%%
color_list = [str(round(x,2)) for x in np.linspace(0.1, 0.9, len(rang))]
#%%
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
for i in range(0, len(gmdat_list)):
    gmdat_list[i].plot(ax=ax, color = color_list[i], marker='s', markersize = 170)
ax.legend(['[{0}, {1})'.format(rang[i], rang[i+1]) for i in range(0, len(rang)-1)],loc='upper left')


#%%
fig.savefig('/Users/mfjkou/sengokuLab/study/LDAT_LS_2019/data/dev/plot/test.png')

#%%
def plot_kriging_res(gmdat, column, rang):
    ''''
    '''
    color_list = [str(round(x,2)) for x in np.linspace(0.1, 0.9, len(rang))]
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    for i in range(0, len(rang)):
        if i< len(rang)-1:
            g = gmdat[(gmdat[column] >= rang[i]) & (gmdat[column] < rang[i+1])]
        else:
            g = gmdat[gmdat[column] >= rang[i]]
        g.plot(ax=ax, color = color_list[i], marker='s', markersize=170)
    ax.legend(['[{0}, {1})'.format(rang[i], rang[i+1]) for i in range(0, len(rang)-1)],loc='best')

    return fig

#%%
plot_kriging_res(gmdat, 'price', rang)

#%% [markdown]
# ## 残差の2乗をプロットする

#%%
gmdat['sq_resid'] = gmodel.krige_residual(mlatlon)**2

#%%
rang = set_rang(gmdat, 'sq_resid', 10)
sq_resid_fig = plot_kriging_res(gmdat, 'sq_resid', rang)

#%%
