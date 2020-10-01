#!/usr/bin/env python3

from argparse import ArgumentParser
import importlib.util
import os


def override_vars(obj):
    for k, v1 in vars(obj).items():
        if k != k.upper():
            continue

        if k not in os.environ:
            continue

        if not isinstance(v1, (int, float, str)):
            continue

        v2 = os.environ[k]
        if isinstance(v1, int):
            v2 = int(v2)
        elif isinstance(v1, float):
            v2 = float(v2)

        print(f'Overriding {k} := {v2}')
        setattr(obj, k, v2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', choices=['train'])
    parser.add_argument('cfg')

    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('configs.config', args.cfg)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    Config = foo.Config

    override_vars(Config)

    cfg = Config()
    cfg.train()


