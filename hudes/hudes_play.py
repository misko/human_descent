import argparse

from hudes.akai_client import AkaiClient
from hudes.keyboard_client import KeyboardClient
from hudes.keyboard_client_openGL import KeyboardClientGL
from hudes.view import OpenGLView, View


def main():
    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--grids", type=int, default=2)

    args = parser.parse_args()

    if args.input == "keyboard":
        controller = KeyboardClient(seed=args.seed, view=View())
    if args.input == "keyboardGL":
        controller = KeyboardClientGL(
            seed=args.seed,
            view=OpenGLView(grid_size=args.grid_size, grids=args.grids),
            step_size_resolution=-0.001,
            inital_step_size_idx=500,
            mesh_grids=args.grids,
            mesh_grid_size=args.grid_size,
        )
    elif args.input == "akai":
        controller = AkaiClient(seed=args.seed, view=View())
    else:
        pass
    controller.run_loop()


if __name__ == "__main__":
    main()
