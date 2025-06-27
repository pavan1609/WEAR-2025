import numpy as np

def random_jitter(X, sigma=0.02):
    return X + np.random.normal(loc=0.0, scale=sigma, size=X.shape)

def random_scaling(X, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * factor

def random_rotation(X):
    if X.shape[2] < 3:
        return X
    theta = np.random.uniform(-np.pi, np.pi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return np.matmul(X, rot_matrix)

def random_channel_dropout(X, drop_prob=0.2):
    X_aug = X.copy()
    for i in range(X.shape[2]):
        if np.random.rand() < drop_prob:
            X_aug[:, :, i] = 0
    return X_aug

def apply_random_augmentations(X):
    if np.random.rand() < 0.5:
        X = random_jitter(X)
    if np.random.rand() < 0.5:
        X = random_scaling(X)
    if np.random.rand() < 0.5:
        X = random_rotation(X)
    if np.random.rand() < 0.5:
        X = random_channel_dropout(X)
    return X
