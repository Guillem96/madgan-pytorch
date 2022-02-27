import typer
from madgan.train import train

_madgan_cli = typer.Typer(name="MAD-GAN CLI")
_madgan_cli.command(train)

if __name__ == "__main__":
    _madgan_cli()