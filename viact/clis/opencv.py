from typer import Typer

app = Typer(
    name="opencv",
    help="Detector using OpenCV",
    no_args_is_help=True,
)


@app.command(name="horizon-detect", help="Run horizon detection")
def detect_horizon(
    image_path: str = "image/three_ships_horizon.JPG",
    with_color_segment: bool = False,
    horizon_estimate_position: float = 1 / 4,
    houghline_thres: int = 200,
):
    from viact.image_io import read_image
    from viact.horizon_detector import HorizonDetectorOpenCV

    image, image_shape = read_image(image_path)
    detector = HorizonDetectorOpenCV(
        with_color_segment=with_color_segment,
        houghline_thres=houghline_thres,
    )
    image = detector(image=image)


@app.command(name="ship-detect", help="Run ship detection")
def detect_ship(
    image_path: str = "image/three_ships_horizone.JPG",
): ...


@app.command(
    name="end-to-end",
    help="Run end-to-end pipeline, including horizon and ship detection",
)
def run_e2e(
    image_path: str = "image/three_ships_horizon.JPG",
    with_color_segment: bool = False,
    horizon_estimate_position: float = 1 / 4,
    houghline_thres: int = 200,
): ...
