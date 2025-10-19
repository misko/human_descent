import argparse
import logging
import os

import pygame as pg

from hudes.controllers.keyboard_client import KeyboardClient
from hudes.controllers.keyboard_client_openGL import KeyboardClientGL
from hudes.controllers.xtouch_client import XTouchClient
from hudes.view import OpenGLView, View


def main(argv=None):

    pg.init()

    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--grids", type=int, default=3)
    parser.add_argument("--addr", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--controller", type=str, default="wireless_osxA")
    parser.add_argument(
        "--skip-help",
        action="store_true",
    )
    parser.add_argument("--max-loop-iterations", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)

    args = parser.parse_args(argv)

    controller = None
    if args.input == "keyboard":
        controller = KeyboardClient(
            addr=args.addr,
            port=args.port,
            seed=args.seed,
            joystick_controller_key=args.controller,
            skip_help=args.skip_help,
        )
        view = View()
    elif args.input == "keyboardGL":
        controller = KeyboardClientGL(
            addr=args.addr,
            port=args.port,
            seed=args.seed,
            step_size_resolution=-0.005,
            inital_step_size_idx=100,
            mesh_grids=args.grids,
            mesh_grid_size=args.grid_size,
            joystick_controller_key=args.controller,
            skip_help=args.skip_help,
        )
        view = OpenGLView(grid_size=args.grid_size, grids=args.grids)
    elif args.input == "xtouch":
        controller = XTouchClient(
            addr=args.addr,
            port=args.port,
            seed=args.seed,
            joystick_controller_key=args.controller,
            skip_help=args.skip_help,
        )
        view = View(use_midi=True)
    else:
        raise ValueError
    try:
        controller.attach_view(view)
        controller.run_loop(
            max_frames=args.max_loop_iterations,
            timeout_s=args.max_seconds,
        )
    finally:
        if controller is not None:
            controller.quit()
            controller.hudes_websocket_client.thread.join(timeout=2.0)
        pg.quit()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
