from typer import Typer
from viact.clis import opencv

app = Typer(name="viact-cv", no_args_is_help=True)
app.add_typer(opencv.app)

if __name__ == "__main__":
    app()
