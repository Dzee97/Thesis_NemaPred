import argparse
from dataclasses import fields, is_dataclass

from snakemake.script import Snakemake


def parse_config[T](config_cls: type[T], snakemake: Snakemake | None) -> T:
    if not is_dataclass(config_cls):
        raise TypeError("Expected a dataclass config definition")

    if snakemake is not None:
        cfg = config_cls(
            **(dict(snakemake.input) | dict(snakemake.output) | dict(snakemake.params))
        )
    else:
        print("Snakemake not in globals")
        parser = argparse.ArgumentParser()
        for field in fields(config_cls):
            flag = f"--{field.name}"
            kwargs = {"type": field.type, "required": True}
            if field.type is bool:
                kwargs = {"action": "store_true", "required": False}

            parser.add_argument(flag, **kwargs)

        cfg = config_cls(**vars(parser.parse_args()))

    return cfg
