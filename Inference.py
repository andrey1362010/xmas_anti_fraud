import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import os
import pickle

print("READ DATA...")
data = pd.read_csv("/data/X_train.csv")

print("LOAD STATISTICS...")
with open('models/global_stat.pickle', 'rb') as handle:
    global_stat = pickle.load(handle)

def get_cat_feature(df, col_name, stat):
    return df[col_name].map(lambda x: stat[col_name].loc[x].result if x in stat[col_name].index else stat[col_name].result.mean()).values

def generate_features(df, stat):
    features = []
    for col_name in ["email","ip","cardToken","paymentSystem",
                     "providerId","bankCountry","partyId","shopId",
                     "currency","bin_hash","ms_pan_hash"]:
        features.append(get_cat_feature(df, col_name, stat))

    features.append(df.amount * df.currency.map({"RUB":1, "USD": 63, "EUR": 63}))

    return np.array(features).T

print("GENERATE FEATURES")
features = generate_features(data, global_stat)

results = []
for model_name in os.listdir("models"):
    print("PREDICT MODEL:", model_name)
    if "model.cbt" not in model_name: continue
    cbc = CatBoostClassifier()
    cbc.load_model(os.path.join("models", model_name))
    res = cbc.predict_proba(features)[:, 1]
    results.append(res)
results = (np.array(results).mean(axis=0) > 0.2)

print("Сохраняем результат...")
data["command_result"] = results
data.to_csv("/data/OUTPUT_CONS.csv", index=False)