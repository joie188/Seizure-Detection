import numpy as np
from cvxopt import solvers
solvers.options['show_progress'] = False
from cvxopt import matrix
import matplotlib.pyplot as plt

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
    C_pos = 0.99*C  
    C_neg = 0.01*C 

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
    #h = np.concatenate((C * np.ones(N), np.zeros(N)))

    pos = np.where(y == 1)
    neg = np.where(y == -1)
    y_C = np.zeros(N)
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
    
    #is_support_vector = (0 < alpha) & (alpha < C)

    is_support_vector = []
    for i in range(len(alpha)):
        if alpha[i] > 0:
            if y[i] == -1:
                value = alpha[i] < C_neg
            elif y[i] == 1:
                value = alpha[i] < C_pos 
        else:
            value = False 
        is_support_vector.append(value)

    y_sv = y[is_support_vector]
    X_sv = X[is_support_vector]

    b = (y_sv - ((alpha * y).reshape(-1, 1) *ker_linear(X, X_sv)).sum(axis=0)).mean()
    
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
    plot_decision_boundary(X_train, y_train, pred_kernel_svm, title='SVM Train')

    validate = np.loadtxt('data/validate_train.csv')
    X_val = validate[:, 0:2]
    y_val = validate[:, 2]
    
    preds = np.array([pred_kernel_svm(x) for x in X_val])
    print('Validation error', (preds * y_val <= 0).mean())
    #plot_decision_boundary(X_val, y_val, pred_kernel_svm, title='SVM Validation')


if __name__ == "__main__": 
    ker = ker_linear
    run_kernel_svm()