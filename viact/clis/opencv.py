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
    num_cluster: int = 4,
    horizon_estimate_y: float = 1 / 4,
    houghline_thres: int = 200,
    verbose: bool = False,
):
    from viact.utils import read_image, save_image
    from viact.horizon_detector import HorizonDetectorOpenCV

    image, image_shape = read_image(image_path)
    detector = HorizonDetectorOpenCV(
        with_color_segment=with_color_segment,
        houghline_thres=houghline_thres,
        verbose=verbose,
    )
    result_dict = detector(
        image=image,
        num_cluster=num_cluster,
        horizon_estimate_y=horizon_estimate_y,
    )

    result_dict = detector.post_process(
        image=image,
        result_dict=result_dict,
    )

    image_postprocess = result_dict["image_postprocess"]

    image_name = f"{image_path.split('/')[-1].split('.')[0]}_horizon"

    save_image(image=image_postprocess, image_name=f"{image_name}.tiff")
    if verbose:
        image_postprocess_grid = result_dict["image_postprocess_grid"]
        save_image(image=image_postprocess_grid, image_name=f"{image_name}_vis.tiff")

    return image_postprocess


@app.command(name="ship-detect", help="Run ship detection")
def detect_ship(
    image_path: str = "image/three_ships_horizon.JPG",
    ship_loc_is_upper: bool = True,
    houghline_thres: int = 200,
    verbose: bool = False,
):
    from viact.utils import read_image, save_image
    from viact.ship_detector import ShipDetectorOpenCV

    image, image_shape = read_image(image_path)
    ship_detector = ShipDetectorOpenCV(verbose=verbose)
    ship_detected_image = ship_detector(
        image,
        houghline_thres,
        ship_loc_is_upper,
    )
    image_name = f"{image_path.split('/')[-1].split('.')[0]}_ship_detected.tiff"
    save_image(image=ship_detected_image, image_name=image_name)


@app.command(
    name="e2e-detect",
    help="Run end-to-end pipeline, including horizon and ship detection",
)
def run_e2e(
    image_path: str = "image/three_ships_horizon.JPG",
    with_color_segment: bool = False,
    num_cluster: int = 4,
    horizon_estimate_y: float = 1 / 4,
    houghline_thres: int = 200,
    verbose: bool = False,
    ship_loc_is_upper: bool = True,
):
    from viact.utils import save_image
    from viact.ship_detector import ShipDetectorOpenCV

    image_postprocess = detect_horizon(
        image_path=image_path,
        with_color_segment=with_color_segment,
        num_cluster=num_cluster,
        horizon_estimate_y=horizon_estimate_y,
        houghline_thres=houghline_thres,
        verbose=verbose,
    )

    ship_detector = ShipDetectorOpenCV(verbose=verbose)
    ship_detected_image = ship_detector(
        image_postprocess,
        houghline_thres,
        ship_loc_is_upper,
    )
    image_name = f"{image_path.split('/')[-1].split('.')[0]}-e2e_res.tiff"
    save_image(image=ship_detected_image, image_name=image_name)
