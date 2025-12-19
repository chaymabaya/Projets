import pandas as pd 
df = pd.read_csv("part3_credit_risk_dirty.csv")
df.columns
df.info()
df.isnull().sum()
df["risk_level"].unique()
df.dropna(subset="risk_level", inplace= True)
df.describe()