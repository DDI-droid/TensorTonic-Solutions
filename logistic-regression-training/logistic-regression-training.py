import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """

    # init
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(steps):
        yp = _sigmoid(X @ w + b)

        dldw = (-1/X.shape[0]) * (np.dot((y - yp), X))

        dldb = (-1/X.shape[0]) * np.sum(y - yp)

        w -= lr * dldw
        b -= lr * dldb
    
    return w, b
    
    