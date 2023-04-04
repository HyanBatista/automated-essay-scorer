from pathlib import Path

import click

from scorer.data import BuildFirstEssaySetDataset


@click.group()
def cli():
    pass


@click.command()
@click.option("--source", type=Path, help="Dados fontes que serão usados para gerar o dataset.")
@click.option("--target", type=Path, help="Local onde o dataset gerado deve ser guardado.")
def build_dataset(source: Path, target: Path):
    BuildFirstEssaySetDataset()(source, target)


@click.command()
@click.option("--data", "-d", type=Path, help="Caminho para os dados a serem utilizados no treino.")
def train(data: Path):
    TrainModel()(data)


cli.add_command(build_dataset)


if __name__ == "__main__":
    cli()
