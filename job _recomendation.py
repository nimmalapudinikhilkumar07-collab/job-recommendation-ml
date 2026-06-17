import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV

df=pd.read_csv("job_data.csv")
print(df.isnull().sum())
df.drop_duplicates(inplace=True)
df.fillna(0,inplace=True)
print(df.columns)
print(df.info())
print(df.describe().T)
df["User_Skills"]=df["User_Skills"].apply(lambda x:x.strip().lower())
df["User_Skills"]=df["User_Skills"].apply(lambda x:x.split(","))

df["Job_Requirements"]=df["Job_Requirements"].apply(lambda x:x.strip().lower())
df["Job_Requirements"]=df["Job_Requirements"].apply(lambda x:x.split(","))
print(df[["User_Skills","Job_Requirements"]])
def matched_skills(job_skills,user_skills):
    count=0
    for skill in user_skills:

        if skill in job_skills:
            count+=1
    return count
df["match_count"]=df.apply(lambda row:matched_skills(row["User_Skills"],row["Job_Requirements"]),axis=1)

df["total_job_skills"]=df["Job_Requirements"].apply(len)
df["match_ratio"]=df["match_count"]/df["total_job_skills"]
print(df["Recommended"].value_counts())
df["total_user_skills"]=df["User_Skills"].apply(len)
df["skill_diff"]=df["total_job_skills"]-df["total_user_skills"]
print(df[["total_job_skills","match_ratio","Match_Score"]].head(20))
def match_score(x,y):

# now training and testing the data with logistic and linear regression

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43,stratify=y)
    model=LogisticRegression(max_iter=1500,class_weight="balanced")
    model.fit(x_train,y_train)
    log_proba=model.predict_proba(x_test)[:,1]
    log_predict=(log_proba>0.5).astype(int)

# all done now move to check with accuracy and all
    print("the model performance with logistic regression")
    print("confusion matrix is :",confusion_matrix(y_test,log_predict))
    print("accuracy is:",accuracy_score(y_test,log_predict))
    print("recall score is :",recall_score(y_test,log_predict))
    print("precision is:",precision_score(y_test,log_predict))
    print("roc auc was:",roc_auc_score(y_test,log_proba))

    print("train score:",model.score(x_train,y_train))
    print("test score:",model.score(x_test,y_test))

#the model is giving slight doubt then checking with random forest and then tuning next and getting best model

    rf_model=RandomForestClassifier(random_state=43,class_weight="balanced")
    rf_model.fit(x_train,y_train)
    rf_proba=rf_model.predict_proba(x_test)[:,1]
    rf_predict=(rf_proba>0.5).astype(int)

    print("the model performance with random forest ")
    print("confusion matrix is :",confusion_matrix(y_test,rf_predict))
    print("accuracy is:",accuracy_score(y_test,rf_predict))
    print("recall score is :",recall_score(y_test,rf_predict))
    print("precision is:",precision_score(y_test,rf_predict))
    print("roc auc was:",roc_auc_score(y_test,rf_proba))

    print("train score:",rf_model.score(x_train,y_train))
    print("test score:",rf_model.score(x_test,y_test))

    rf=RandomForestClassifier(random_state=43,class_weight="balanced")
    param_dist={
        "n_estimators":[20,50,100],
        "max_depth":[10,20,40],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4]
    }
    rf_tuned=RandomizedSearchCV(estimator=rf,param_distributions=param_dist,n_iter=5,n_jobs=-1,cv=5,scoring="recall",random_state=43)
    rf_tuned.fit(x_train,y_train)

    best_model=rf_tuned.best_estimator_
    best_model.fit(x_train,y_train)
    best_proba=best_model.predict_proba(x_test)[:,1]
    best_predict=(best_proba>0.5).astype(int)


    print("the model performance with best model ")
    print("confusion matrix is :",confusion_matrix(y_test,best_predict))
    print("accuracy is:",accuracy_score(y_test,best_predict))
    print("recall score is :",recall_score(y_test,best_predict))
    print("precision is:",precision_score(y_test,best_predict))
    print("roc auc was:",roc_auc_score(y_test,best_proba))

    print("train score:",best_model.score(x_train,y_train))
    print("test score:",best_model.score(x_test,y_test))


    print(df["Recommended"].value_counts())

    importance=best_model.feature_importances_
    feature=x.columns

    feature_df=pd.DataFrame({
        "Feature":feature,
        "importance":importance
    })
    print(feature_df.sort_values(by="importance",ascending=False))

print("we are checking with match score feature and without match score")
print("with match score ")
match_score(df[["match_count","total_job_skills","match_ratio","skill_diff","total_user_skills","Match_Score"

        ]],df["Recommended"])

print("now without match score :")
match_score(df[["match_count","total_job_skills","match_ratio","skill_diff","total_user_skills"

        ]],df["Recommended"])
