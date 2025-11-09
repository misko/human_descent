# Human Descent Architecture

## System Overview
Human Descent couples an interactive browser (or native pygame) client with a Python inference server. Users explore a random low-dimensional subspace of a trained MNIST model by nudging parameters through keyboard, mouse, or MIDI controls. Every input generates protobuf messages over WebSockets, and the server responds with updated losses, predictions, mesh grids, and leaderboard data. The project is organized as two top-level roots:

- `hudes/`: Python backend, realtime server, protobuf schema, and native controllers.
- `web/`: Vite/Three.js/Chart.js frontend, browser controllers, and Playwright tests.

The sections below document the runtime flow, stateful components, MVC split, and the protobuf socket contract that binds both halves.

## Backend Runtime (Python, `hudes/`)
### Core modules
| Module | Responsibility |
| --- | --- |
| `websocket_server.py` | Async WebSocket server, HTTP health endpoints, per-client state machine, and orchestration of inference workers. |
| `model_data_and_subspace.py` | Loads MNIST models/datasets, fuses parameters, samples random subspaces, and runs training/validation/mesh computations. |
| `websocket_client.py` | Aggregates outgoing control commands for native clients and manages the socket thread. |
| `controllers/*` & `hudes_play.py` | Local pygame entry points that attach physical inputs to the backend via `HudesWebsocketClient`. |
| `high_scores.py` | SQLite helpers powering leaderboard APIs. |

### Execution pipeline
1. `run_server()` (`hudes/websocket_server.py`) creates a listener for WebSocket clients and an HTTP API server for health/leaderboard queries.
2. For each WebSocket connection, `process_client()` instantiates a `Client` dataclass that tracks configuration, queued parameter deltas, timers, and help state.
3. `inference_runner()` spawns a separate process or thread (`listen_and_run`) that owns a `ModelDataAndSubspace` instance, fuses model parameters, and services batched requests from every client via multiprocessing queues.
4. The async loop `inference_runner_clients()` polls clients, packages requests (`train`, `val`, `mesh`, `loss_line`, `sgd`), and pushes them to the worker queue.
5. `inference_result_sender()` pulls worker results, encodes protobuf `Control` messages (`hudes_pb2.py`), and writes them back on each client's socket.
6. Auxiliary coroutines schedule speed-run deadlines, emit batch example snapshots, and expose leaderboard APIs.

### Client dataclass state (`hudes/websocket_server.py:227`)
| Field | Purpose |
| --- | --- |
| `next_step` / `current_step` | Accumulates pending dimension deltas until they are applied in the worker loop. |
| `dims_offset`, `dims_at_a_time` | Track which slice of the random subspace the user controls right now. |
| `batch_idx`, `batch_size` | Control dataset sampling on both train and validation splits. |
| `dtype`, `mesh_grid_size`, `mesh_step_size`, `mesh_grids` | Configure inference precision and visualization payloads. |
| `sgd`, `total_sgd_steps`, `force_reset_weights` | Queue explicit SGD micro-steps and manage cached per-client weight tensors. |
| `speed_run_active`, `speed_run_end_time`, `speed_run_log`, `best_val_loss_during_run`, `high_score_logged` | Manage timed “speed run” competitions and persist their inputs. |
| `request_idx`, `active_request_idx`, `active_inference`, `sent_batch` | Provide back-pressure so that every protobuf reply references the user-visible request counter. |

### Backend state transitions
1. **Configuration** (`CONTROL_CONFIG`): updates dtype, dims, batch size, mesh options, and seeds; forces an inference refresh.
2. **Dimension step** (`CONTROL_DIMS`): merges deltas into `next_step`. When `client_runner_q` wakes up it promotes them to `current_step`, requests `train` + optional `mesh` jobs, and resets `next_step`.
3. **Batch advance** (`CONTROL_NEXT_BATCH`): increments `batch_idx`, forces a fresh batch example payload, and schedules `train`+`val` passes so charts/leaderboards stay in sync.
4. **New subspace** (`CONTROL_NEXT_DIMS`): bumps `dims_offset`, zeros `current_step`, and reseeds `ModelDataAndSubspace.get_dim_vec()` so the player explores a new random projection.
5. **SGD micro-step** (`CONTROL_SGD_STEP`): if not inside a speed run, queues `sgd` iterations so `listen_and_run()` calls `mad.sgd_step()` and persists the returned weights.
6. **Speed run** (`CONTROL_SPEED_RUN_START` / automatic timeout / `CONTROL_HIGH_SCORE_LOG`): resets cached weights, starts a countdown, logs every inbound control before the deadline, and requests a final validation. When that `VAL` arrives it transitions to `AwaitingScore` until the user submits a high-score record.
7. **Disconnect**: closing the socket removes the client’s entry from `active_clients` and cancels its scheduled tasks; cached weights are released the next time `listen_and_run()` sees the missing key.

### Inference worker (`listen_and_run`)
- Maintains a `client_weights` tensor clone per client ID, seeded from `ModelDataAndSubspace.saved_weights`.
- Applies `mad.delta_from_dims()` to map the sparse `current_step` dict into a dense delta vector before every train/val/mesh job.
- Provides hooks for additional visualizations: mesh grids (`get_loss_grid`) and loss lines (`get_loss_lines`).
- Ensures dtype-specific copies stay in sync by moving results back to `float32` when needed.

## Frontend Runtime (web/)
### Boot sequence (`web/app.js`)
1. Detect backend host/port via query params, build-time env vars, or health probes.
2. Choose render mode (`1d` keyboard landscape vs `3d` GL) and instantiate `KeyboardClient` or `KeyboardClientGL` with options such as `meshGrids`, `rowSpacing`, and debug flags.
3. Expose the client on `window.__hudesClient` for manual inspection and tests, install UI toggles, then call `client.runLoop()`.

### Browser client core (`web/client/HudesClient.js`)
| Concern | Behavior |
| --- | --- |
| **Socket lifecycle** | Opens `ws://host:port`, sets `binaryType='arraybuffer'`, and defers all outbound traffic through `sendQ()` so protobuf.js can encode payloads once the schema loads. |
| **State** | Owns a `ClientState` instance for step size, dtype, batch size, help screens, timer flags, and formatting helpers for the HUD. |
| **View binding** | Creates a `ViewRouter` that currently wraps `LandscapeView` (Three.js scene + Chart.js HUD). The view instance is responsible for mesh grids, charts, loss lines, and help overlays. |
| **Event handling** | Registers `keydown/keyup`, debounces repeats, and routes commands to helper methods (`sendDimsAndSteps`, `getNextBatch`, `getNextDims`, `getSGDStep`, `startSpeedRun`, `submitHighScore`). |
| **Message pump** | The `_handleMessage` switch decodes protobuf payloads, reshapes arrays, updates charts, keeps loss history arrays aligned, and maintains a local countdown mirror for speed runs. |

### Browser controller variants
- `KeyboardClient` maps paired keys to dimension deltas (alt layouts supported) and manages repeat timers so each press sends batched `CONTROL_DIMS` payloads.
- `KeyboardClientGL` disables paired keys, enabling WASD/mouse-driven navigation of individual grids, rotation, and scroll-powered steps. It installs `mouseControls.js` for drag/scroll gestures and maps them into parameter updates in the selected grid.
- Future controllers (mouse-only, touchscreen, gamepads) can extend `HudesClient` in the same way by overriding `initInput()` and `processKeyPress()` but reusing the networking/view stack.

### Client state container (`web/client/ClientState.js`)
| Field | Purpose |
| --- | --- |
| `stepSizeIdx`, `stepSizeResolution`, `stepSize`, `minLogStepSize`, `maxLogStepSize` | Maintain logarithmic step sizes with bracket keys. |
| `batchSizes`, `batchSizeIdx`, `batchSize` | Rotate through supported batch sizes. |
| `dtypes`, `dtypeIdx`, `dtype` | Toggle precision hints sent to the backend. |
| `bestScore`, `sgdSteps` | Track progress and annotate the HUD. |
| `helpScreenFns`, `helpScreenIdx` | Drive overlay images for tutorials. |
| `speedRunActive`, `speedRunSecondsRemaining` | Mirror authoritative server fields for local rendering. |

### Client-side state transitions
1. **Initialization**: `ClientState` sets defaults, `HudesClient.runLoop()` blocks until protobuf definitions load, then sends `CONTROL_CONFIG`.
2. **User input**: Keyboard or mouse events call `sendDimsAndSteps`, `getNextBatch`, etc., which encode protobuf payloads and optimistically update local buffers (e.g., `dimsAndStepsOnCurrentDims`).
3. **Server messages**: Each protobuf reply updates history arrays, charts, and HUD annotations. Validation messages call `ClientState.updateBestScoreOrNot`, and mesh/loss-line payloads mutate Three.js meshes.
4. **Speed run**: `startSpeedRun()` toggles timers and waits for server countdown data; the state returns to normal only after a `CONTROL_VAL_LOSS` message arrives with `speedRunFinished=true`.
5. **Help overlay**: Text input is ignored while `helpScreenIdx >= 0`; cycling help screens eventually resumes normal event handling.

## MVC Mapping
| Layer | Implementation | Notes |
| --- | --- | --- |
| **Model** | On the backend, `ModelDataAndSubspace` plus per-client weight clones implement the data/model state. On the frontend, `ClientState` mirrors the user-adjustable subset (step size, dtype, help). |
| **View** | `LandscapeView` renders mesh grids, charts, HUD overlays, and sample digits; it exposes methods (`updateLossChart`, `updateMeshGrids`, `highlightLossLine`, etc.) so controllers stay passive. Native pygame views (`hudes/view.py`, `hudes/opengl_view.py`) follow the same contract. |
| **Controller** | `KeyboardClient*` and `XTouchClient` translate hardware events into protobuf control messages. The backend controller loop (`process_client`) likewise interprets protobuf commands and transitions `Client` state, acting as the other half of MVC. |

Code flow summary:
1. **Input (Controller)**: User presses keys → controller computes a `dim -> delta` map.
2. **State update (Model)**: Controller enqueues a protobuf `CONTROL_DIMS`; backend `Client.next_step` collects it; inference worker applies deltas to weight tensors.
3. **Rendering (View)**: Worker responses are streamed back as protobuf `Control` messages; the browser view updates visuals while the backend can display pygame overlays for local builds.

## WebSocket + Protobuf Socket
### Transport layer
- **Browser**: `HudesClient` opens a WebSocket (`binaryType='arraybuffer'`) and uses `protobuf.js` (via `ProtoLoader.js`) to encode/decode the shared `hudes.proto` schema (`web/public/hudes.proto`, generated copies live under `web/dist/` and `hudes/hudes_pb2.py`).
- **Native controllers**: `HudesWebsocketClient` (`hudes/websocket_client.py`) runs in a background thread, accumulates many small `CONTROL_DIMS` payloads into a single message per animation frame, and multiplexes send/receive queues for pygame clients.

### Message catalog
| Direction | Type | Payload highlights |
| --- | --- | --- |
| Client → Server | `CONTROL_CONFIG` | `Config` seed, dims, mesh config, batch size, dtype, mesh toggle, loss lines. |
|  | `CONTROL_DIMS` | Repeated `{dim, step}` entries plus `request_idx`. Aggregated client-side to reduce chatter. |
|  | `CONTROL_NEXT_DIMS` | Requests a fresh random subspace; server resets offsets. |
|  | `CONTROL_NEXT_BATCH` | Advances `batch_idx`, causing new sample previews and loss recomputation. |
|  | `CONTROL_SGD_STEP` | Increments `sgd` counter (ignored when `speed_run_active`). |
|  | `CONTROL_SPEED_RUN_START` | Begins timed competition; server snapshots inbound controls. |
|  | `CONTROL_HIGH_SCORE_LOG` | Sends a sanitized 4-character name to persist the current score. |
| Server → Client | `CONTROL_BATCH_EXAMPLES` | Flattened tensors + shapes for `n` MNIST samples. |
|  | `CONTROL_TRAIN_LOSS_AND_PREDS` | Scalar loss, flattened predictions, confusion matrix, `total_sgd_steps`, optional countdown seconds. |
|  | `CONTROL_VAL_LOSS` | Validation loss, countdown, and `speed_run_finished` flag when appropriate. |
|  | `CONTROL_MESHGRID_RESULTS` | Flattened loss landscapes + shape metadata. |
|  | `CONTROL_LOSS_LINE_RESULTS` | 1D slices for each controlled dimension (1D mode). |
|  | `CONTROL_LEADERBOARD_RESPONSE` | Parallel arrays of names/scores. |

### Typical sequence
1. **Config/handshake**
   - Client boots, loads proto, sends `CONTROL_CONFIG`.
   - Server replies with `CONTROL_BATCH_EXAMPLES`, `CONTROL_TRAIN_LOSS_AND_PREDS`, `CONTROL_MESHGRID_RESULTS` based on initial seeds.
2. **Interactive step**
   - User taps a key → controller builds `{dim: delta}` → `sendDimsAndSteps()` enqueues `CONTROL_DIMS`.
   - Python server merges deltas into `next_step`, schedules a `train` job, and returns `CONTROL_TRAIN_LOSS_AND_PREDS` followed by (optionally) `CONTROL_MESHGRID_RESULTS`.
3. **Speed run**
   - Client sends `CONTROL_SPEED_RUN_START` and suppresses SGD.
   - Server resets weights, sets `speed_run_end_time`, and includes `speed_run_seconds_remaining` in every outbound message.
   - When timer elapses, server flags `request_full_val`, emits a final `CONTROL_VAL_LOSS` with `speed_run_finished=true`, and waits for `CONTROL_HIGH_SCORE_LOG`.

### Socket helpers
- `HudesWebsocketClient.send_config()` (Python) and `HudesClient.sendConfig()` (browser) both wrap protobuf creation so controllers remain declarative.
- The browser `sendQ()` waits for `readyState === OPEN`, assigns `requestIdx`, verifies the payload against the protobuf schema, and logs every send when debug mode is enabled.
- The native socket loop coalesces multiple `CONTROL_DIMS` entries before calling `websocket.send()`; this mirrors the browser-side repeat handling and keeps latency low even with high-frequency key presses.

## References
- Backend source: `hudes/websocket_server.py`, `hudes/model_data_and_subspace.py`, `hudes/websocket_client.py`.
- Frontend source: `web/app.js`, `web/client/HudesClient.js`, `web/client/KeyboardClient*.js`, `web/client/views/LandscapeView.js`, `web/client/ClientState.js`.
- Shared protocol: `hudes/hudes.proto`, `web/public/hudes.proto` (compiled to `hudes/hudes_pb2.py`).
