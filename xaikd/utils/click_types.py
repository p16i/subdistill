import click
import typing
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
    output_type = pathlib.Path

    def convert(self, value, param, ctx):
        if isinstance(value, self.output_type):
            return value

        return pathlib.Path(value)


class List(click.ParamType):
    output_type = typing.List[str]

    def convert(self, value, param, ctx):
        if isinstance(value, typing.List):
            return value

        return value.split(",")


class SmartFloat(click.ParamType):
    def convert(self, value, param, ctx):
        # the type provides a way to specify Float with expression, e.g., 1-0.1
        return eval(value)
