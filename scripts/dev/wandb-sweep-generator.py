import click

import os
from pathlib import Path

from glob import glob
import yaml

import wandb

WANDB_SWEEP_URL = f"https://wandb.ai/<ENTITY>/<PROJECT>/sweeps/<SWEEP_ID>"


@click.command()
@click.option("--wandb-entity", type=str, default="xaikd")
@click.option("--wandb-project", type=str, required=True)
@click.option("--dry-run", type=bool, is_flag=True)
@click.option("--config-files", type=str, required=True)
def main(wandb_entity, wandb_project, dry_run, config_files):

    if dry_run:
        click.echo(f"--- [dry-run={dry_run}] ----")

    arr_config_files = sorted(glob(config_files))
    click.echo(f"Found {len(arr_config_files)} config files!")
    click.echo("\n".join(list(map(lambda f: f" - {f}", arr_config_files))))

    click.echo("------")

    for config_file in arr_config_files:
        path = Path(config_file)
        filename = path.stem
        folder_name = Path(os.path.split(path)[-2]).stem

        with open(config_file, "r") as fh:
            sweep_config = yaml.safe_load(fh)
            sweep_config["name"] = f"{folder_name}/{filename}"

            sweep_group = f"{folder_name}"

            sweep_config["parameters"]["wandb-experiment-group"] = dict(
                value=sweep_group
            )
            total_runs = 1
            for k, v in sweep_config["parameters"].items():
                assert isinstance(v, dict)

                if "value" in v:
                    assert not isinstance(v["value"], list), f"key={k} is failed!"
                elif "values" in v:
                    assert isinstance(v["values"], list), f"key={k} is failed!"
                    total_runs = total_runs * len(v["values"])

            if not dry_run:

                print("====== output from wandb.sdk ======")
                sweep_id = wandb.sweep(
                    entity=wandb_entity,
                    project=wandb_project,
                    sweep=sweep_config,
                )
                print("====== end ======")

            else:
                sweep_id = "dry-run-dummy-id"

        url = (
            WANDB_SWEEP_URL.replace("<ENTITY>", wandb_entity)
            .replace("<PROJECT>", wandb_project)
            .replace("<SWEEP_ID>", sweep_id)
        )

        click.echo(f"- Sweep from `{folder_name}/{filename}`")
        click.echo(
            f"\t- {wandb_entity}/{wandb_project}/{sweep_id} (total {total_runs} runs) ({url})"
        )


if __name__ == "__main__":
    main()
