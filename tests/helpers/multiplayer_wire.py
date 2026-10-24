"""Python half of the JavaScript/Python gameplay wire compatibility check."""

import base64
import json
import sys

from google.protobuf.json_format import MessageToDict, ParseDict

from hudes.hudes_pb2 import Control


def main():
    request = json.loads(sys.argv[1])
    decoded = {}
    encoded = {}
    for name, fixture in request["fixtures"].items():
        expected = ParseDict(fixture, Control())
        actual = Control.FromString(base64.b64decode(request["encoded"][name]))
        # Comparing protobuf messages also checks explicitly-present generation=0.
        if actual != expected:
            raise AssertionError(f"JavaScript encoding differs for {name}: {actual}")
        decoded[name] = MessageToDict(actual, use_integers_for_enums=True)
        encoded[name] = base64.b64encode(expected.SerializeToString()).decode("ascii")
    print(json.dumps({"encoded": encoded, "decoded": decoded}))


if __name__ == "__main__":
    main()
