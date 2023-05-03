import click

import pathlib

from xaikd import models, datasets


class Model(click.ParamType):
    def convert(self, value, param, ctx):
        return models.get_model(value)


class DatasetConfiguration(click.ParamType):
    def convert(self, value, param, ctx):
        return datasets.construct(value)


class Path(click.ParamType):
    def convert(self, value, param, ctx):
        return pathlib.Path(value)
