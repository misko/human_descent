import argparse

from hudes.akai_client import AkaiClient
from hudes.keyboard_client import KeyboardClient
from hudes.keyboard_client_openGL import KeyboardClientGL
from hudes.view import OpenGLView, View


def main():
    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.input == "keyboard":
        controller = KeyboardClient(seed=args.seed, view=View())
    if args.input == "keyboardGL":
        controller = KeyboardClientGL(seed=args.seed, view=OpenGLView(grid_size=51))
    elif args.input == "akai":
        controller = AkaiClient(seed=args.seed, view=View())
    else:
        pass
    controller.run_loop()


if __name__ == "__main__":
    main()
