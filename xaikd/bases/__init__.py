import typing
import numpy as np
import numpy.typing as npt
from abc import ABC


class Basis(ABC):
    def __init__(self, centering: bool = True):
        self.centering = centering

    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def __str__(self) -> str:
        prefix = type(self).__name__.lower()

        suffix = "centered" if self.centering else "uncentered"

        return "-".join([prefix, suffix])


class PCA(Basis):
    def fit(self, activation: np.ndarray, context: np.ndarray):
        """_summary_

        Args:
            activation (np.ndarray): _description_
            context (np.ndarray): We do NOT use this here!
        """

        n, d = activation.shape

        if self.centering:
            mean = np.mean(activation, axis=0)
        else:
            mean = np.zeros(d)

        activation = activation - mean
        eigvals, eigvecs = np.linalg.eigh(activation.T @ activation / n)

        index = np.arange(d)[::-1]

        eigvals = eigvals[index]
        eigvecs = eigvecs[:, index]

        return eigvecs, mean, eigvals


class PRCA(Basis):
    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_ Summary

        Args:
            activation (np.ndarray): _description_
            context (np.ndarray): _description_

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        n, d = activation.shape

        if self.centering:
            mean = np.mean(activation, axis=0)
        else:
            mean = np.zeros(d)

        activation = activation - mean
        eigvals, eigvecs = np.linalg.eigh(
            ((activation.T @ context + context.T @ activation)) / n
        )

        index = np.arange(d)[::-1]

        eigvals = eigvals[index]
        eigvecs = eigvecs[:, index]

        return eigvecs, mean, eigvals


class PRCAAbs(Basis):
    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().fit(activation, context)


class PRCARelRecon(Basis):
    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().fit(activation, context)
