import argparse

from hudes.akai_client import AkaiClient
from hudes.keyboard_client import KeyboardClient


def main():
    parser = argparse.ArgumentParser(description="Hudes: Keyboardclient")
    parser.add_argument("--input", type=str, default="keyboard")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.input == "keyboard":
        controller = KeyboardClient(seed=args.seed)
    elif args.input == "akai":
        controller = AkaiClient(seed=args.seed)
    else:
        pass
    controller.run_loop()


if __name__ == "__main__":
    main()
