import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from algorithm import gradient_descent_multi


df = pd.read_csv("part1_pricing_gradient_descent_dirty.csv")
df.describe()
df.columns
df.shape
df.duplicated().sum()
df.isnull().sum()
len(df['demand_index'][df['demand_index'] < 0])
df.columns


columns = [
    'demand_index', 'time_slot', 'day_of_week', 'competition_pressure',
    'operational_cost', 'seasonality_index', 'marketing_intensity',
    'dynamic_price'
]

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(columns):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

# replace value that depace the limits with clipping
df["time_slot"] = df["time_slot"].clip(lower=0, upper=23)
df["day_of_week"] = df["day_of_week"].clip(lower=0, upper=6)


# (df.time_slot[df.time_slot > 23]).count()
# df = df[(df["time_slot"] >= 0) & (df["time_slot"] <= 23)]
# df.shape

# (df.day_of_week[df.day_of_week > 6]).count()
# df = df[(df["day_of_week"] >= 0) & (df["day_of_week"] <= 6)]
# df.shape


cols_replace_negatives = ["demand_index", "operational_cost", "marketing_intensity", "competition_pressure"] 
# for col in cols_replace_negatives: 
#     df[col] = df[col].apply(lambda x: pd.NA if x < 0 else x)

df[cols_replace_negatives] = df[cols_replace_negatives].where(df[cols_replace_negatives] >= 0, pd.NA)

df.dropna(subset=["dynamic_price"], inplace=True)
df.shape

mean_data = ["operational_cost", "seasonality_index"]
median_data = ["demand_index" , "time_slot" ,"day_of_week" , "competition_pressure", "marketing_intensity" ]

for col in mean_data:
    df[col] = df[col].fillna(df[col].mean())

for col in median_data:
    df[col] = df[col].fillna(df[col].median())


cleaned_data = df.copy()
cleaned_data.to_csv("cleaned_data.csv", index=False)

# dff = pd.read_csv("cleaned_data.csv")
# dff.shape

X = df.drop("dynamic_price" , axis=1 )
columns_X = X.columns


y = df["dynamic_price"]
columnds_y = y.name

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0 , 1))
X_scaled = scaler.fit_transform(X)
df1 = pd.DataFrame(X_scaled, columns= columns_X, index= X.index)

df1.to_csv("scaled_data.csv", index=False)


# sns.pairplot(df1)
# plt.show()

for col in X.columns:
    plt.figure(figsize=(5,4))
    plt.scatter(X[col], y, alpha=0.6)
    plt.xlabel(col)
    plt.ylabel("dynamic_price")
    plt.title(f"{col} vs dynamic_price")
    plt.show()


gradient_descent_multi(X_scaled , y, method='batch')
gradient_descent_multi(X_scaled , y, method='stochastic')
gradient_descent_multi(X_scaled , y, method='mini-batch')

for method in ['batch', 'stochastic', 'mini-batch']:
    print(f"\nUsing {method} gradient descent:")
    theta, loss_history = gradient_descent_multi(X_scaled , y, method=method, lr=0.01, iterations=1000, batch_size=32, thresh=0.1)
    print(f"Final theta values for {method} GD: {theta[1:]}, Intercept: {theta[0]}")

# theta = gradient_descent_multi(X_scaled , y)
# for i in range(len(theta)):
#     if i == 0:
#         print(f"Intercept: {theta[i]:.4f}")
#     else:
#         print(f"Theta {i}: {theta[i]:.4f}")
# print(f"Final theta values: {theta[1:]}, Intercept: {theta[0]}")
theta.shape
# gradient_descent_multi(X_scaled , y)df
import pandas as pd
import numpy as np  
X_scaled = pd.read_csv("scaled_data.csv")
y = pd.read_csv("part1_pricing_gradient_descent_dirty.csv")["dynamic_price"]
X_scaled.shape
y.shape

df.describe()