import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

train_df = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/titanic_dataset/train.csv")
test_df = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/titanic_dataset/test.csv")
submission = pd.read_csv("C:/Users/327ae/OneDrive/바탕 화면/py/titanic_dataset/gender_submission.csv")

# drop_first => 첫번째 열은 제외 / 하나가 1이면 무조건 다른 하나는 0일 때
train_df_corr = pd.get_dummies(train_df, columns=["Embarked","Sex"])

train_corr = train_df_corr.corr()
# reset_index => 행 번호 다시 설정, drop=True => 원래 행 번호 삭제
all_df = pd.concat([train_df,test_df], sort=False).reset_index(drop=True)

Fare_mean = all_df[["Pclass","Fare"]].groupby("Pclass").mean().reset_index()
Fare_mean.columns = ["Pclass","Fare_mean"]

all_df = pd.merge(all_df,Fare_mean,on="Pclass",how="left")
all_df.loc[(all_df["Fare"].isnull()),"Fare"]=all_df["Fare_mean"]
all_df = all_df.drop("Fare_mean",axis=1)

name_df = all_df["Name"].str.split("[.,]",2,expand=True)
name_df.columns = ["family_name","honorific","name"]

name_df["family_name"] = name_df["family_name"].str.strip()
name_df["honorific"] = name_df["honorific"].str.strip()
name_df["name"] = name_df["name"].str.strip()

all_df = pd.concat([all_df,name_df],axis=1)
train_df = pd.concat([train_df,name_df[0:len(train_df)].reset_index(drop=True)],axis=1)
test_df = pd.concat([test_df,name_df[len(train_df):].reset_index(drop=True)],axis=1)

honorific_df = train_df[["honorific","Survived","PassengerId"]].dropna().groupby(["honorific","Survived"]).count().unstack()
honorific_df.plot.bar(stacked=True)

honorific_age_mean = all_df[["honorific","Age"]].groupby("honorific").mean().reset_index()
honorific_age_mean.columns = ["honorific","honorific_Age"]

all_df = pd.merge(all_df, honorific_age_mean, on="honorific", how="left")
all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["honorific_Age"]
all_df = all_df.drop(["honorific_Age"],axis=1)

all_df["family_num"]=all_df["Parch"] + all_df["SibSp"]

all_df.loc[all_df["family_num"] ==0, "alone"] = 1
all_df["alone"].fillna(0, inplace=True)

all_df = all_df.drop(["PassengerId","Name","family_name","name","Ticket","Cabin"],axis=1)

categories = all_df.columns[all_df.dtypes == "object"]

all_df.loc[~((all_df["honorific"] =="Mr") |
    (all_df["honorific"] =="Miss") |
    (all_df["honorific"] =="Mrs") |
    (all_df["honorific"] =="Master")), "honorific"] = "other"

# 결측치 출력
# print(all_df.isnull().sum())

all_df["Embarked"].fillna("missing", inplace=True)

for cat in categories:
    le = LabelEncoder()
    if all_df[cat].dtypes == "object":    
        le = le.fit(all_df[cat])
        all_df[cat] = le.transform(all_df[cat])
        
train_X = all_df[~all_df["Survived"].isnull()].drop("Survived",axis=1).reset_index(drop=True)
train_Y = train_df["Survived"]
test_X = all_df[all_df["Survived"].isnull()].drop("Survived",axis=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2)

categories = ["Embarked", "Pclass", "Sex","honorific","alone"]
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
lgb_eval = lgb.Dataset(X_valid, y_valid,  categorical_feature=categories, reference=lgb_train)

lgbm_params = {
    "objective":"binary",        
    "random_seed":1234,
}

# model_lgb = lgb.train(lgbm_params, 
#                       lgb_train, 
#                       valid_sets=lgb_eval, 
#                       num_boost_round=100,
#                       early_stopping_rounds=5,
#                       verbose_eval=10)

# y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

from sklearn.metrics import accuracy_score

# print(accuracy_score(y_valid, np.round(y_pred)))

folds = 3

kf = KFold(n_splits=folds)

models = []

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
        
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
    lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=categories, reference=lgb_train)    
    
    model_lgb = lgb.train(lgbm_params, 
                          lgb_train, 
                          valid_sets=lgb_eval, 
                          num_boost_round=100,
                          early_stopping_rounds=20,
                          verbose_eval=10,
                         )
    
    
    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    
    models.append(model_lgb)

preds = []

for model in models:
    pred = model.predict(test_X)
    preds.append(pred)
    
preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)

preds_int = (preds_mean > 0.5).astype(int)

submission["Survived"] = preds_int

submission.to_csv("C:/Users/327ae/OneDrive/바탕 화면/py/titanic_dataset/titanic_submit01.csv",index=False)

submission.to_csv()
