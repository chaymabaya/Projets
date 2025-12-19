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



columns = df.columns
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

def batch_descent_multi(X, y, lr=0.01, iterations=1000, thresh=0.1):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  
    print(X.shape)
    theta = np.zeros(n + 1)
    loss_history = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        gradient = (2/m) * X.T.dot(y_pred - y) # Calcul de la dirivée de la fonction de MSE
                                                # gradient[0] → intercept update
                                                # gradient[1:] → coefficients update
        theta -= lr * gradient
        MSE = np.mean((y - y_pred)**2) # mean = sum of squared errors / number of samples
        loss_history.append(MSE)
        # if i % 100 == 0:
        #     print(f"Iteration {i}, MSE {MSE}")
        if MSE < thresh:
            break
    return theta , pd.DataFrame(loss_history)



def stochastic_gradient_descent_multi(X, y, lr=0.01, iterations=1000, thresh=0.1):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  
    theta = np.zeros(n + 1)
    y = np.array(y)
    loss_history = []
    for i in range(iterations):
        for j in range(m):
            y_pred = X[j].dot(theta)
            error = y_pred - y[j]
            gradient = 2 * X[j] * error
            theta -= lr * gradient
        y_pred_full = X.dot(theta)
        MSE = np.mean((y - y_pred_full)**2)
        loss_history.append(MSE)
        # if i % 100 == 0:
        #     print(f"Iteration {i}, MSE {MSE}")
        if MSE < thresh:
            break
    return theta , loss_history


def mini_batch_gradient_descent_multi(X, y, lr=0.01, iterations=1000, batch_size=32, thresh=0.1):
    m , n = X.shape
    X = np.c_[np.ones(m), X]
    theta = np.zeros(n + 1)
    loss_history = []
    for i in range(iterations): 
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y.iloc[indices]
        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            y_pred = X_batch.dot(theta)
            gradient = (2/len(y_batch)) * X_batch.T.dot(y_pred - y_batch)
            theta -= lr * gradient
        y_pred_full = X.dot(theta)
        MSE = np.mean((y - y_pred_full)**2)
        loss_history.append(MSE)
        # if i % 100 == 0:
        #     print(f"Iteration {i}, MSE {MSE}")
        if MSE < thresh:
            break
    return theta , loss_history


def gradient_descent_multi(X, y, method='batch', lr=0.01, iterations=1000, batch_size=32, thresh=0.1):
    if method == 'batch':
        return batch_descent_multi(X, y, lr, iterations, thresh=thresh)
    elif method == 'stochastic':
        return stochastic_gradient_descent_multi(X, y, lr, iterations, thresh=thresh)
    elif method == 'mini-batch':
        return mini_batch_gradient_descent_multi(X, y, lr, iterations, batch_size, thresh=thresh)
    else:
        raise ValueError("Invalid method. Choose 'batch', 'stochastic', or 'mini-batch'.")
    
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# print("Train shape:", X_train.shape, y_train.shape)
# print("Test shape:", X_test.shape, y_test.shape)



gradient_descent_multi(X_scaled , y, method='batch')
gradient_descent_multi(X_scaled , y, method='stochastic')
gradient_descent_multi(X_scaled , y, method='mini-batch')

for method in ['batch', 'stochastic', 'mini-batch']:
    print(f"\nUsing {method} gradient descent:")
    theta, loss_history = gradient_descent_multi(X_scaled , y, method=method, lr=0.01, iterations=1000, batch_size=32, thresh=0.1)
    print(f"Final theta values for {method} GD: {theta[1:]}, Intercept: {theta[0]}")


def compare_models(X, y):
    methods = {
        "Batch": batch_descent_multi,
        "Stochastic": stochastic_gradient_descent_multi,
        "Mini-Batch": mini_batch_gradient_descent_multi
    }
    
    plt.figure(figsize=(10,6))
    
    for name, func in methods.items():
        if name == "Mini-Batch":
            theta, loss_history = func(X, y, lr=0.01, iterations=500, batch_size=32)
        else:
            theta, loss_history = func(X, y, lr=0.01, iterations=500)
        plt.plot(loss_history, label=name)
    
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Comparison of Gradient Descent Variants")
    plt.legend()
    plt.show()


compare_models(X_scaled, y)

import matplotlib.pyplot as plt

# Define learning rates to test
learning_rates = [0.001, 0.005, 0.01, 0.05]

# Store MSE histories for each method and learning rate
mse_histories_batch = {}
mse_histories_stochastic = {}
mse_histories_minibatch = {}

# Run each method for each learning rate
for lr in learning_rates:
    _, loss_batch = batch_descent_multi(X_scaled, y, lr=lr, iterations=500)
    _, loss_stochastic = stochastic_gradient_descent_multi(X_scaled, y, lr=lr, iterations=500)
    _, loss_minibatch = mini_batch_gradient_descent_multi(X_scaled, y, lr=lr, iterations=500, batch_size=32)
    
    mse_histories_batch[lr] = loss_batch
    mse_histories_stochastic[lr] = loss_stochastic
    mse_histories_minibatch[lr] = loss_minibatch

# -----------------------------
# Plot for Batch Gradient Descent
# -----------------------------
plt.figure(figsize=(10,6))
for lr in learning_rates:
    plt.plot(mse_histories_batch[lr], label=f"lr={lr}")
plt.xlabel("Itérations")
plt.ylabel("MSE")
plt.title("Impact du Learning Rate sur la convergence (Batch Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Plot for Stochastic Gradient Descent
# -----------------------------
plt.figure(figsize=(10,6))
for lr in learning_rates:
    plt.plot(mse_histories_stochastic[lr], label=f"lr={lr}")
plt.xlabel("Itérations")
plt.ylabel("MSE")
plt.title("Impact du Learning Rate sur la convergence (Stochastic Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Plot for Mini-Batch Gradient Descent
# -----------------------------
plt.figure(figsize=(10,6))
for lr in learning_rates:
    plt.plot(mse_histories_minibatch[lr], label=f"lr={lr}")
plt.xlabel("Itérations")
plt.ylabel("MSE")
plt.title("Impact du Learning Rate sur la convergence (Mini-Batch Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()



import pickle
with open("gradient_descent_models.pkl", "wb") as f:
    pickle.dump({
        "batch": batch_descent_multi(X_scaled, y, lr=0.01, iterations=1000),
        "stochastic": stochastic_gradient_descent_multi(X_scaled, y, lr=0.01, iterations=1000),
        "mini-batch": mini_batch_gradient_descent_multi(X_scaled, y, lr=0.01, iterations=1000, batch_size=32)
    }, f)

