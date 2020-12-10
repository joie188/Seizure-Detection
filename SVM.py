import numpy as np
import pandas as pd
from cvxopt import solvers
solvers.options['show_progress'] = False
from cvxopt import matrix
import matplotlib.pyplot as plt


def train_kernel_svm(X, y, k=None, C=1, gamma=1):
    """
    inputs
    X: data matrix, shape (N, d)
    y: label matrix, shape (N,)
    k: if k is None, then compute the kernel by XX^T; else k could be a function or precomputed matrix
    C: coefficient of slack terms in primal optimization, scalar 
    
    returns
    w: weight, shape (N,)
    b: bias, scalar
    """
    N = len(y)
    if callable(k):
        K = k(X, X, gamma)
    elif k is None:
        k = ker_linear
        K = X.dot(X.T)
    else:
        K = k

    print("hello")
    P = K.reshape(N, N) * y.reshape(N, 1).dot(y.reshape(1, N))
    q = -np.ones(N)

    G = np.concatenate((np.eye(N), -np.eye(N)))
    # h = np.concatenate((C * np.ones(N), np.zeros(N)))
    
    C_pos = (np.count_nonzero(y == 1)/N)*C  
    C_neg = (np.count_nonzero(y == -1)/N)*C 

    y_C = np.zeros(N)
    pos = np.where(y == 1)
    neg = np.where(y == -1)
    y_C[pos] = C_pos 
    y_C[neg] = C_neg 
    h = np.concatenate((y_C, np.zeros(N)))
    
    A = y.reshape(1, N)
    A = A.astype('float')
    b = np.zeros(1)
    
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alpha = np.array(sol['x'])
    alpha[alpha < 1e-4] = 0
    alpha = alpha.reshape(-1)
    
    # is_support_vector = (0 < alpha) & (alpha < C)

    is_support_vector = []
    for i in range(len(alpha)):
        value = False
        if alpha[i] > 0:
            if y[i] == -1:
                value = abs(alpha[i] - C_neg) <= 0.01
            elif y[i] == 1:
                value = abs(alpha[i] - C_pos) <= 0.01 
        is_support_vector.append(value)

    y_sv = y[is_support_vector]
    X_sv = X[is_support_vector]
    b = (y_sv - ((alpha * y).reshape(-1, 1)*k(X, X_sv, gamma)).sum(axis=0)).mean()
    return alpha, b

def get_pred_kernel_svm(alpha, b, X, y, ker, gamma=1):
    return lambda x: (alpha * y * ker(X, x, gamma).reshape(-1)).sum(axis=0) + b

# KERNELS

ker_linear = lambda X1, X2: X1.dot(X2.T)

def kernel_poly(x, y, p=2):				
	return (np.dot(x, y)) ** p	

def kernel_rbf(X1, X2, gamma):
    if len(X1.shape) == 1:
        X1 = X1.reshape(1, -1) # (N1, d)
    if len(X2.shape) == 1:
        X2 = X2.reshape(1, -1) # (N2, d)
    X1 = np.expand_dims(X1, axis=1) # (N1, 1, d)
    X2 = np.expand_dims(X2, axis=0) # (1, N2, d)
    
    # broadcasting trick
    return np.exp(-gamma * np.sum((X1 - X2) ** 2, axis=2))

# From plotBoundary.py
def plot_decision_boundary(X, Y, scoreFn, contour_values=[-1, 0, 1], title=""):
    """
    Plot the decision boundary. For that, we asign a score to
    each point in the mesh [x_min, m_max] x [y_min, y_max].

    X is data matrix (each row is a data point)
    Y is desired output (1 or -1)
    scoreFn is a function of a data point
    contour_values is a list of values to plot
    """

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max - x_min) / 200.0, (y_max - y_min) / 200.0)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)

    plt.figure()
    CS = plt.contour(
        xx, yy, zz, contour_values, colors="green", linestyles="solid", linewidths=2
    )
    plt.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=(1.0 - Y).ravel(), s=50, cmap=plt.cm.cool)
    plt.title(title)
    plt.axis("tight")
    plt.show()


def run_kernel_svm(ker=ker_linear, C=1):
    print('hi')
    train = pd.read_csv('data/train_data.csv').values 
    X_train = train[:, :-1].copy()
    #X_train = train[:, [1,3,4,5,6,7,10,19,20,21,22,23]].copy()
    y_train = train[:, -1].copy()
    print('hi')

    validate = pd.read_csv('data/val_data.csv').values
    X_val = validate[:, :-1].copy()
    y_val = validate[:, -1].copy()

    test = pd.read_csv('data/test_data.csv')
    X_test = test.values[:, :-1].copy()
    y_test = test.values[:, -1].copy()


    for C in [1]:
        for gamma in [0.01]:
            alpha, b = train_kernel_svm(X_train, y_train, k=ker, C=C, gamma=gamma)
            print(alpha, b)
            pred_kernel_svm = get_pred_kernel_svm(alpha, b, X_train, y_train, ker, gamma=gamma)
            print('C', C) 
            print('Gamma', gamma)
            #w = np.sum(np.array([alpha * y_train]).T * X_train, axis = 0)
            #print('W Indices:', np.argsort(abs(w)))

            preds = np.array([pred_kernel_svm(x) for x in X_train])
            print('Training error', (preds * y_train <= 0).mean())

            preds = np.array([pred_kernel_svm(x) for x in X_val])
            print('Validation error', (preds * y_val <= 0).mean())

            preds = np.array([pred_kernel_svm(x) for x in X_test])
            print('Test error', (preds * y_test <= 0).mean())

if __name__ == "__main__": 
    ker = kernel_rbf
    run_kernel_svm(ker=ker)