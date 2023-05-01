import os
import typing
import numpy as np
import numpy.typing as npt

import torch
from abc import ABC

from . import learners


BASES = dict()


def register_basis(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""
        BASES[name] = cls

        return cls

    return wrapped


class Basis(ABC):
    def __init__(self, alias, centering: bool = True):
        self.centering = centering
        self.alias = alias
        self.artifact: dict

    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def __str__(self) -> str:
        prefix = self.alias

        suffix = "centered" if self.centering else "uncentered"

        return "-".join([prefix, suffix])

    def save(self, output_dir: str):
        if hasattr(self, "artifact") is None:
            raise ValueError("Artifact is NONE! Please fit first.")
        output_dir = f"{output_dir}/{self}"

        os.makedirs(output_dir, exist_ok=True)

        for k, v in self.artifact.items():
            np.save(f"{output_dir}/{k}", v)


def get_basis(name, **kwargs) -> Basis:
    return BASES[name](alias=name, **kwargs)


@register_basis("pca")
class PCA(Basis):
    artifact_keys = ["eigvecs", "mean", "eigvals"]

    def fit(self, activation: np.ndarray, context: np.ndarray, **kwargs):
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

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, mean, eigvals)))

        return eigvecs, mean, eigvals


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "mean", "eigvals"]

    def fit(
        self, activation: np.ndarray, context: np.ndarray, **kwargs
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

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, mean, eigvals)))

        return eigvecs, mean, eigvals


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs", "mean"]
    mode: str

    def fit(
        self, activation: np.ndarray, context: np.ndarray, **kwargs
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

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        U = learner.fit(activation, context, **kwargs)

        self.artifact = dict(zip(self.artifact_keys, (U, mean)))

        return U, mean, None


@register_basis("prca-abs")
class PRCAAbs(PRCAVariant):
    mode = "abs"


@register_basis("prca-recon")
class PRCARelRecon(Basis):
    mode = "recon"
