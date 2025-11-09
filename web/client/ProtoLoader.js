import protobuf from 'protobufjs';

// Vite will replace import.meta.env.VITE_BUILD_ID at build time if defined
const BUILD_ID = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_BUILD_ID)
  ? String(import.meta.env.VITE_BUILD_ID)
  : '';

export async function loadProto(protoPath) {
  // Prevent stale caching by appending a build id query when available
  const url = BUILD_ID ? `${protoPath}?v=${encodeURIComponent(BUILD_ID)}` : protoPath;
  const root = await protobuf.load(url);
  const Control = root.lookupType('hudes.Control');
  const ControlType = root.lookupEnum('hudes.Control.Type');
  return { Control, ControlType, root };
}
