import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.3.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier
import xgboost as xgb

def featureEngineering(df):
    #df = df[df['price'] < 20000]
    df.loc[df['bedrooms'] > 6,'bedrooms'] = 6
    df.loc[df['bathrooms'] > 6,'bathrooms'] = 6

    df["num_photos"] = df["photos"].apply(len)
    df["num_features"] = df["features"].apply(len)
    df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["created_hour"] = df["created"].dt.hour
    #df['bedrooms_bathrooms'] = df['bedrooms'] * 10.0 + df['bathrooms']
    df['feat_elevator'] = df['features'].map(lambda x: 'Elevator' in x)
    df['feat_animals_allowed'] = df['features'].map(lambda x: ('Cats Allowed' in x) or ('Dogs Allowed' in x))
    df['feat_hardwood_floor'] = df['features'].map(lambda x: ('Hardwood Floors' in x) or ('HARDWOOD' in x))
    df['feat_doorman'] = df['features'].map(lambda x: 'Doorman' in x)
    df['feat_dishwasher'] = df['features'].map(lambda x: 'Dishwasher' in x)
    df['feat_no_fee'] = df['features'].map(lambda x: 'No Fee' in x)
    df['feat_laundry_unit'] = df['features'].map(lambda x: ('Laundry in Unit' in x))
    df['feat_laundry_building'] = df['features'].map(lambda x: ('Laundry in Building' in x))
    df['feat_fit_center'] = df['features'].map(lambda x: 'Fitness Center' in x)
    df['feat_pre_war'] = df['features'].map(lambda x: ('Pre-War' in x) or ('prewar' in x))
    df['feat_roof_deck'] = df['features'].map(lambda x: 'Roof Deck' in x)
    df['feat_outdoor_space'] = df['features'].map(lambda x: ('Outdoor Space' in x) or ('Common Outdoor Space' in x))
    df['feat_pool'] = df['features'].map(lambda x: 'Swimming Pool' in x)
    df['feat_new_construction'] = df['features'].map(lambda x: 'New Construction' in x)
    df['feat_terrace'] = df['features'].map(lambda x: 'Terrace' in x)
    df['feat_loft'] = df['features'].map(lambda x: 'Loft' in x)
    df['washer'] = df['features'].map(lambda x: ('washer' in x) or ('Washer' in x))
    df['parking'] = df['features'].map(lambda x: ('parking' in x) or ('Parking' in x))
    df['internet'] = df['features'].map(lambda x: ('internet' in x) or ('Internet' in x))
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
    df['distance'] = pointTopoint((df.latitude,df.longitude),bronx_center)
    #df.loc[df['distance'] > 3,'distance'] = 10

train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")

train_df = train_df
test_df = test_df

features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price","num_photos","num_features",\
                    "created_year","created_month","created_day","created_hour","feat_elevator","feat_animals_allowed","feat_hardwood_floor",\
                    "feat_doorman","feat_dishwasher","feat_no_fee","feat_laundry_unit","feat_laundry_building",\
                    "feat_fit_center","feat_pre_war","feat_roof_deck","feat_outdoor_space","feat_pool","feat_new_construction",\
                    "feat_terrace","feat_loft","washer","parking","internet","distance"]
target_num_map = {'high':0, 'medium':1, 'low':2}


featureEngineering(train_df)

featureEngineering(test_df)
train_X = train_df[features_to_use]
test_X = test_df[features_to_use]

train_Y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

x_train, x_test, y_train, y_test = train_test_split(train_X,train_Y,test_size=0.3)


extraT = ExtraTreesClassifier(n_estimators=1000)
# extraT.fit(x_train,y_train)
#print(log_loss(y_test,extraT.predict_proba(x_test)))

rf1 = RandomForestClassifier(n_estimators=1000,criterion='gini')
rf2 = RandomForestClassifier(n_estimators=1000,criterion='entropy')
rf3 = RandomForestClassifier(n_estimators=1000,criterion='gini',max_features="log2")
# rf.fit(x_train,y_train)
# print(log_loss(y_test,rf.predict_proba(x_test)))

gra = GradientBoostingClassifier(n_estimators=1000)
# gra.fit(x_train,y_train)
# print(log_loss(y_test,gra.predict_proba(x_test)))

b = xgb.XGBClassifier(n_estimators=200)
b1 = xgb.XGBClassifier(n_estimators=200,booster = "gblinear")

model = VotingClassifier(estimators=[("xgb",b),("xgb",b1)],voting='soft')#
#model = VotingClassifier(estimators=[('rf',rf1),('rf2',rf2),('rf3',rf3) ,('gra',gra),("xgb",b)],voting='soft')
model.fit(x_train,y_train)
print(log_loss(y_test,model.predict_proba(x_test)))



pred = model.predict_proba(test_X)

res = pd.DataFrame(pred,columns = ['high','medium','low'],index=test_df.listing_id)
res.to_csv("files/voting.csv")


