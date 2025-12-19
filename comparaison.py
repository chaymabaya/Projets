import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 1. Génération des données simulées
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Ajout du biais
X_b = np.c_[np.ones((100, 1)), X]

# Paramètres initiaux
theta_init = np.random.randn(2, 1)
alpha = 0.1
n_epochs = 50

# 2. Batch Gradient Descent
def batch_gradient_descent(X, y, theta, alpha, n_epochs):
    m = len(y)
    history = []
    for epoch in range(n_epochs):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - alpha * gradients
        loss = mean_squared_error(y, X.dot(theta))
        history.append(loss)
    return theta, history

# 3. Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, theta, alpha, n_epochs):
    m = len(y)
    history = []
    for epoch in range(n_epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index+1]
            yi = y[rand_index:rand_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - alpha * gradients
        loss = mean_squared_error(y, X.dot(theta))
        history.append(loss)
    return theta, history

# 4. Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, theta, alpha, n_epochs, batch_size=20):
    m = len(y)
    history = []
    for epoch in range(n_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = 2/len(xi) * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - alpha * gradients
        loss = mean_squared_error(y, X.dot(theta))
        history.append(loss)
    return theta, history

# 5. Entraînement avec chaque méthode
theta_batch, loss_batch = batch_gradient_descent(X_b, y, theta_init.copy(), alpha, n_epochs)
theta_sgd, loss_sgd = stochastic_gradient_descent(X_b, y, theta_init.copy(), alpha, n_epochs)
theta_minibatch, loss_minibatch = mini_batch_gradient_descent(X_b, y, theta_init.copy(), alpha, n_epochs, batch_size=20)

# 6. Affichage des résultats
print("Batch GD -> intercept:", theta_batch[0][0], "weight:", theta_batch[1][0])
print("SGD -> intercept:", theta_sgd[0][0], "weight:", theta_sgd[1][0])
print("Mini-Batch GD -> intercept:", theta_minibatch[0][0], "weight:", theta_minibatch[1][0])

# 7. Visualisation des courbes de coût
plt.plot(loss_batch, label="Batch GD")
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_minibatch, label="Mini-Batch GD")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Comparaison des méthodes de Gradient Descent")
plt.show()