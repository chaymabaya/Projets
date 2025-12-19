

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
    