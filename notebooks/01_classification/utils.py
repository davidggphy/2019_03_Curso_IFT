from sklearn.datasets import make_circles


def generate_concentric_dataset(n_points=200):
    X, Y = make_circles(n_points, noise=0.15, factor=0.5, random_state=0)
    X[:, 0] *= 3
    X = [3, 4] + X
    return X, Y
