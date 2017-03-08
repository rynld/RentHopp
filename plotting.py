from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']



feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]



train_df = pd.read_json("input/train.json")

def violinBedroomsBathrooms(train_df):
    train_df['bathrooms'].ix[train_df['bathrooms'] <= 0] = 1
    train_df['bedrooms'].ix[train_df['bedrooms'] <= 0] = 1

    train_df['bathrooms'].ix[train_df['bathrooms'] > 3] = 3
    train_df['bedrooms'].ix[train_df['bedrooms'] > 3] = 3

    train_df['bedrooms_bathrooms'] = train_df['bedrooms'] * 10.0 + train_df['bathrooms']
    print(train_df['bedrooms_bathrooms'])
    # plt.figure(figsize=(8,4))

    sns.violinplot(x='interest_level', y='bedrooms_bathrooms', data=train_df)
    plt.show()
    
def barBedroomsBathrooms(df):
    df['bathrooms'].ix[df['bathrooms'] <= 0] = 1
    df['bedrooms'].ix[df['bedrooms'] <= 0] = 1

    df['bathrooms'].ix[df['bathrooms'] > 3] = 3
    df['bedrooms'].ix[df['bedrooms'] > 3] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    sns.countplot(x='bedrooms_bathrooms', hue='interest_level',data=df)
    plt.show()

def barBedroomsBathroomsPrice(df):
    df.loc[df['bathrooms'] <= 0,'bathrooms'] = 1
    df.loc[df['bedrooms'] <= 0,'bedrooms'] = 1

    df.loc[df['bathrooms'] > 3,'bathrooms'] = 3
    df.loc[df['bedrooms'] > 3,'bedrooms'] = 3

    df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']

    df = df[df['price'] < 10000]

    trans = {
        'low':'red',
        'medium': 'green',
        'high':'blue'
    }
    colors = [trans[x.interest_level] for i,x in df.iterrows()]


    plt.scatter(df.price, df.bedrooms_bathrooms, c=colors)

    #plt.yticks(np.arange(0,40,1))
    plt.yticks([11,12,13,14,15,21,22,23,24,31,32,33,34])
    plt.show()

def location(df, distance = 3):
    def pointTopoint(a,b):
        a = (np.radians(a[0]),np.radians(a[1]))
        b = (np.radians(b[0]), np.radians(b[1]))

        dlon = b[1] - a[1]
        dlat = b[0] - a[0]
        a = np.power(np.sin(dlat / 2), 2) + np.cos(a[0]) * np.cos(b[0]) * np.power((np.sin(dlon / 2)), 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 3961 * c
        return d

    bronx_center = (40.834600, -73.879082)
    extreme_point = (40.859532, -73.913929)

    #distance_from_center = pointTopoint(bronx_center,extreme_point)

    df['distance'] = pointTopoint((df.latitude,df.longitude),bronx_center)
    df = df[pointTopoint(bronx_center,(df.latitude,df.longitude)) <= distance]

    return df

def mapa(df):
    map = Basemap(projection='merc', lat_0=mlat, lon_0=mlon, resolution='l', area_thresh=1.0,
                       llcrnrlon=-num, llcrnrlat=num,
                       urcrnrlon=-num, urcrnrlat=num)

    map.readshapefile('/home/person/zipfolder/rds/tl_2010_48_prisecroads', 'Streets', drawbounds=False)

    for shape in map.Streets:
        xx, yy, = zip(*shape)
        map.plot(xx, yy, linewidth=1.5, color='green', alpha=.75)
        ##Same for zip codes
    # df = df.head(10)
    # p = (40.859532, -73.913929)
    # map = Basemap(lon_0=p[0],lat_0=p[1])
    #
    # plt.figure(figsize=(19,20))
    # map.bluemarble()
    # for i,x in df.iterrows():
    #     a,b = map(x.longitude,x.latitude)
    #     map.plot(a,b,marker='o',color='red',markersize=5)
    # plt.show()

def correlation(df):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True, n=100)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
                square=True, xticklabels=True, yticklabels=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()



correlation(train_df)
