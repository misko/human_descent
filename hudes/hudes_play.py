import argparse
import logging
import os

# from hudes.akai_client import AkaiClient
from hudes.keyboard_client import KeyboardClient
from hudes.keyboard_client_openGL import KeyboardClientGL
from hudes.view import OpenGLView, View
from hudes.xtouch_client import XTouchClient


def main():
    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--grids", type=int, default=2)

    args = parser.parse_args()

    if args.input == "keyboard":
        controller = KeyboardClient(seed=args.seed)
        view = View()
    elif args.input == "keyboardGL":
        controller = KeyboardClientGL(
            seed=args.seed,
            step_size_resolution=-0.005,
            inital_step_size_idx=100,
            mesh_grids=args.grids,
            mesh_grid_size=args.grid_size,
        )
        view = OpenGLView(grid_size=args.grid_size, grids=args.grids)
    # elif args.input == "akai":
    #     controller = AkaiClient(seed=args.seed)
    #     view = View()
    elif args.input == "xtouch":
        controller = XTouchClient(seed=args.seed)
        view = View(use_midi=True)
    else:
        raise ValueError
    controller.attach_view(view)
    controller.run_loop()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
