from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


def MyEL():
    return BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
