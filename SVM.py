import numpy as np
from cvxopt import solvers
#solvers.options['show_progress'] = False
from cvxopt import matrix
import matplotlib.pyplot as plt

def train_kernel_svm(X, y, k=None, C=1):
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
        K = k(X, X)
    elif k is None:
        K = X.dot(X.T)
    else:
        K = k

    P = K.reshape(N, N) * y.reshape(N, 1).dot(y.reshape(1, N))
    q = -np.ones(N)

    G = np.concatenate((np.eye(N), -np.eye(N)))
    h = np.concatenate((C * np.ones(N), np.zeros(N)))

    A = y.reshape(1, N)
    b = np.zeros(1)
    
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alpha = np.array(sol['x'])
    alpha[alpha < 1e-4] = 0
    alpha = alpha.reshape(-1)
    
    is_support_vector = (0 < alpha) & (alpha < C)
    y_sv = y[is_support_vector]
    X_sv = X[is_support_vector]
    b = (y_sv - ((alpha * y).reshape(-1, 1) * ker(X, X_sv)).sum(axis=0)).mean()
    
    return alpha, b

def get_pred_kernel_svm(alpha, b, X, y, ker):
    return lambda x: (alpha * y * ker(X, x).reshape(-1)).sum(axis=0) + b


ker_linear = lambda X1, X2: X1.dot(X2.T)

def kernel_poly(x, y, p):				
	return (np.dot(x, y)) ** p	
ker_poly = lambda X1, X2: ker_poly(X1, X2, p=2)

def kernel_rbf(X1, X2, gamma):
    if len(X1.shape) == 1:
        X1 = X1.reshape(1, -1) # (N1, d)
    if len(X2.shape) == 1:
        X2 = X2.reshape(1, -1) # (N2, d)
    X1 = np.expand_dims(X1, axis=1) # (N1, 1, d)
    X2 = np.expand_dims(X2, axis=0) # (1, N2, d)
    
    # broadcasting trick
    return np.exp(-gamma * np.sum((X1 - X2) ** 2, axis=2))
ker_rbf = lambda X1, X2: ker_rbf(X1, X2, gamma=1)


def run_kernel_svm(ker=ker_linear, C=1):
    train = np.loadtxt('data/data_train.csv')
    X_train = train[:, 0:2].copy()
    y_train = train[:, 2].copy()

    alpha, b = train_kernel_svm(X_train, y_train, k=ker, C=C)
    pred_kernel_svm = get_pred_kernel_svm(alpha, b, X_train, y_train, ker)

    preds = np.array([pred_kernel_svm(x) for x in X_train])
    print('Training error', (preds * y_train <= 0).mean())
    #plot_decision_boundary(X_train, y_train, pred_kernel_svm, title='SVM Train')

    validate = np.loadtxt('data/validate_train.csv')
    X_val = validate[:, 0:2]
    y_val = validate[:, 2]
    
    preds = np.array([pred_kernel_svm(x) for x in X_val])
    print('Validation error', (preds * y_val <= 0).mean())
    #plot_decision_boundary(X_val, y_val, pred_kernel_svm, title='SVM Validation')


if __name__ == "__main__":  
    ker = ker_linear
    run_kernel_svm()