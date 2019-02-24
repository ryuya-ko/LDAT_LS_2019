#%%
import folium
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pykrige as pk

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
# - pykrige packageのRegression kriginはsklearnのモデルのみ受け取る
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
y.head()

#%%
from pykrige.rk import RegressionKriging
gmodel = RegressionKriging(regression_model=model,  variogram_model='exponential', nlags=15, coorinates_type = )

#%%
gmodel.fit(prop, latlon, y)

#%%
my = gmodel.predict(mprop, mlatlon)

#%%
pred_list = []
for i in range(0,len(dat)+1):
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

#%%
pred_list

#%%
# RMSE
y - pred_list


#%%
pred_list.flatten()

#%%
