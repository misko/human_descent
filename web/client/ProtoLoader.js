import protobuf from 'protobufjs';

export async function loadProto(protoPath) {
    const root = await protobuf.load(protoPath);
    const Control = root.lookupType('hudes.Control');
    const ControlType = root.lookupEnum('hudes.Control.Type');
    return { Control, ControlType };
  }
