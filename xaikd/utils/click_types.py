import click

import pathlib

from xaikd import models
from xaikd import datasets


class Model(click.ParamType):
    def convert(self, value, param, ctx):
        return models.get_trained_model(value)


class DatasetConfiguration(click.ParamType):
    def convert(self, value, param, ctx):
        return datasets.construct(value)


class Path(click.ParamType):
    def convert(self, value, param, ctx):
        return pathlib.Path(value)


class List(click.ParamType):
    def convert(self, value, param, ctx):
        return value.split(",")


class SmartFloat(click.ParamType):
    def convert(self, value, param, ctx):
        # the type provides a way to specify Float with expression, e.g., 1-0.1
        return eval(value)
