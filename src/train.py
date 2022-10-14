from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from pathlib import Path
from joblib import dump
import pandas as pd
from sklearn.metrics import accuracy_score 
import os

def main():
  data = pd.read_csv("../data/customer_churn.csv")
  n = round(len(data)*0.8)
  if not os.path.exists("../data/train"): os.makedirs("../data/train")
  if not os.path.exists("../data/val"): os.makedirs("../data/val")
      
  data[:n].to_csv("../data/train/train.csv", index = False)
  data[n:].to_csv("../data/val/val.csv", index = False)
  train = "../data/train/train.csv"
  val = "../data/val/val.csv"
  train = pd.read_csv(train)
  val = pd.read_csv(val)

  train = pd.get_dummies(train, columns=['gender', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod'] )

  val = pd.get_dummies(val, columns=['gender', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod'] )


  train.replace({"Churn":{'Yes':1, 'No':0}}, inplace=True)
  val.replace({"Churn":{'Yes':1, 'No':0}}, inplace=True)

  train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0).astype(float)
  val['TotalCharges'] = pd.to_numeric(val['TotalCharges'], errors='coerce').fillna(0).astype(float)

  X_train = train.drop('customerID', axis=1)
  X_val = val.drop('customerID', axis=1)

  y_train = train['Churn']
  y_val = val['Churn']

  lasso = Lasso(alpha=0.5)
  trained_model = lasso.fit(X_train.values, y_train.values)
  dump(trained_model, "../model/model.joblib")

  accuracy = accuracy_score(y_val, [round(x) for x in trained_model.predict(X_val.values)])
  metrics = "accuracy" , 

  with open('../metrics/metric.txt', 'w') as f:
    f.write(str(accuracy))


if __name__ == "__main__":
    main()
