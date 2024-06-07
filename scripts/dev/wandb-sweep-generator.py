import click

import os
from pathlib import Path

from glob import glob
import yaml

import wandb

WANDB_USERNAME = "p16i"
WANDB_SWEEP_URL = f"https://wandb.ai/{WANDB_USERNAME}/<PROJECT>/sweeps/<SWEEP_ID>"


@click.command()
@click.option("--wandb-project", type=str, required=True)
@click.option("--dry-run", type=bool, is_flag=True)
@click.option("--config-files", type=str, required=True)
def main(wandb_project, dry_run, config_files):

    if dry_run:
        click.echo(f"--- [dry-run={dry_run}] ----")

    arr_config_files = glob(config_files)
    click.echo(f"Found {len(arr_config_files)} config files!")
    click.echo("\n".join(list(map(lambda f: f" - {f}", arr_config_files))))

    click.echo("------")

    for config_file in arr_config_files:
        filename = Path(config_file).stem

        click.echo(f"- Sweep from `{filename}`")
        if not dry_run:
            with open(config_file, "r") as fh:
                sweep_config = yaml.safe_load(fh)
                sweep_config["name"] = filename

                sweep_id = wandb.sweep(
                    project=wandb_project,
                    sweep=sweep_config,
                )

        else:
            sweep_id = "dry-run-dummy-id"

        url = WANDB_SWEEP_URL.replace("<PROJECT>", wandb_project).replace(
            "<SWEEP_ID>", sweep_id
        )

        click.echo(f"\t- {WANDB_USERNAME}/{wandb_project}/{sweep_id} ({url})")


if __name__ == "__main__":
    main()
