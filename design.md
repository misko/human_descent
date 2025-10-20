# Human Descent – Web Speed Run Design

## Overview
This document defines the design of the **Speed Run** mode and leaderboard system for both the **web client** and **backend server** of *Human Descent*.
It focuses on:
- Protocol (protobuf) message types, inputs/outputs, and field mappings
- State machines for server and client
- Main runtime loops and message-driven updates
- Known ambiguities and follow-ups

It complements `high_score.md` by clarifying client/server contracts and describing the code-accurate behavior.

---

## 1. Protocol Summary
**File:** `hudes/hudes.proto`

### Envelope: `Control`
- `type` (enum `Control.Type`) — message role
- `request_idx` (optional int32) — sequential index for correlation

### Core Payloads
**Client → Server**
- `CONTROL_CONFIG`: `config`
- `CONTROL_DIMS`: `dims_and_steps[]`
- `CONTROL_NEXT_BATCH`
- `CONTROL_NEXT_DIMS`
- `CONTROL_SGD_STEP`: `sgd_steps` (ignored during Speed Run)
- `CONTROL_SPEED_RUN_START`
- `CONTROL_HIGH_SCORE_LOG`: `{ name }` (server validates name strictly; score computed server-side)
- `CONTROL_QUIT`

**Server → Client**
- `CONTROL_BATCH_EXAMPLES`: `{ type, n, train_data, train_labels, shapes, batch_idx }`
- `CONTROL_TRAIN_LOSS_AND_PREDS`: `{ train_loss, preds, confusion_matrix, total_sgd_steps, speed_run_seconds_remaining? }`
- `CONTROL_VAL_LOSS`: `{ val_loss, speed_run_seconds_remaining?, speed_run_finished? }`
- `CONTROL_MESHGRID_RESULTS`: `{ mesh_grid_results[], mesh_grid_shape[], speed_run_seconds_remaining? }`

### Notes
- Wire uses `snake_case`; JS via `protobuf.js` exposes `lowerCamelCase` (e.g., `speedRunSecondsRemaining`, `speedRunFinished`).
- Countdown is included while a run is active in TRAIN/VAL/MESH messages.
- End-of-run is signaled by `CONTROL_VAL_LOSS` with `speed_run_finished=true`.
- The deprecated `CONTROL_FULL_LOSS` is removed; do not use.

---

## 2. Server Design
**File:** `hudes/websocket_server.py`

### 2.1 Core Components
Each client has a state object containing:
- **Runtime controls:** steps, batch, dims, dtype, mesh config, SGD counters
- **Scheduling:** flags for updates, requests, and force states
- **Speed Run:** timer, log, best validation loss, and flags

An async inference worker handles compute. Two loops manage scheduling and sending.

### 2.2 Main Loops
1. **`inference_runner_clients`** — schedules next op (`train`, `mesh`, `val`, `sgd`).
2. **`inference_result_sender`** — serializes results into protobuf `Control` messages and sends to client.

### 2.3 Client → Server Messages
| Message | Action |
|----------|---------|
| CONFIG | Updates configuration and may trigger `force_update` |
| DIMS | Applies delta to weights, schedules train/mesh |
| NEXT_DIMS | Increments dims offset |
| NEXT_BATCH | Increments batch, triggers validation |
| SGD_STEP | Increments SGD (ignored during Speed Run) |
| SPEED_RUN_START | Resets weights, starts timer, clears logs |
| HIGH_SCORE_LOG | Finalizes run, persists results |
| QUIT | Closes client loop |

### 2.4 Server → Client Messages
| Type | Description |
|------|--------------|
| BATCH_EXAMPLES | Sends sample data and labels |
| TRAIN_LOSS_AND_PREDS | Sends loss, predictions, confusion matrix |
| VAL_LOSS | Sends validation loss |
| MESHGRID_RESULTS | Sends grid data and remaining time during Speed Run |

### 2.5 Scheduling Logic
- `force_update=True` → triggers next operation.
- Validation runs (`VAL_LOSS`) occur in normal cadence and at Speed Run finalization.
- SGD steps run only when not in Speed Run.
- Worker maintains per-client cloned weights; resets via `force_reset_weights`.
- At the Speed Run deadline, the server schedules a final `VAL_LOSS`; in `inference_result_sender`, that VAL is marked with `speed_run_finished=true` and the server deactivates `speed_run_active`.

---

## 3. Client Design
**File:** `web/client/HudesClient.js`

### 3.1 Initialization
- Detects `ws/wss` target host/port.
- Loads protobuf (`loadProto('hudes.proto')`).
- Instantiates `View` + `ClientState`.
- Sends initial `CONTROL_CONFIG`.

### 3.2 User Inputs
| Key | Function |
|-----|-----------|
| R | Start Speed Run |
| Delete/Backspace | Single SGD step (disabled during run) |
| Space | Next dims |
| Enter | Next batch |
| [ / ] | Adjust step size |
| ; | Toggle batch size |
| ' | Toggle dtype |
| X | Help overlay |

### 3.3 Client → Server Messages
Follows JS `lowerCamelCase` mapping:
- `config`, `dimsAndSteps`, `nextDims`, `nextBatch`, `sgdStep`, `speedRunStart`, `highScoreLog`.

### 3.4 Server → Client Handlers
- **TRAIN_LOSS_AND_PREDS:** updates loss, preds, confusion matrix.
- **VAL_LOSS:** updates validation loss, triggers best-score check.
- **BATCH_EXAMPLES:** updates displayed samples.
- **MESHGRID_RESULTS:** updates view and Speed Run countdown.

### 3.5 Speed Run UX
- **Start:** User presses **R** → sends `CONTROL_SPEED_RUN_START`.
- **During:**
	- Server includes `speed_run_seconds_remaining` in TRAIN/VAL/MESH while active.
	- Client starts a local countdown on first server countdown and updates the HUD every ~300ms using wall time; UI shows seconds with 1 decimal precision.
	- SGD is disabled during a run (client guard; server ignores).
- **End:** The run ends only when a `CONTROL_VAL_LOSS` arrives with `speed_run_finished=true`. The client then prompts for a 4-character alphanumeric name and includes the achieved final VAL loss in the prompt. It submits `CONTROL_HIGH_SCORE_LOG` once. No local end occurs purely from countdown reaching 0.

---

## 4. State Machines

### 4.1 Server State (per client)
| State | Description | Transition |
|--------|--------------|-------------|
| Normal | Idle | Start run |
| SpeedRunActive | Timer running | Deadline reached → schedule final VAL |
| AwaitingFinalVAL | Waiting for VAL completion | Send VAL with `speed_run_finished=true` → Normal |
| HighScoreLogged | Finalized | Restart via new run |

**Guards & Side Effects**
- Ignore SGD during runs.
- Append incoming messages to `speed_run_log` only before the strict deadline.
- Include countdown in TRAIN/VAL/MESH results while active.
- At end-of-run VAL, set `speed_run_finished=true` and deactivate `speed_run_active` server-side.
- Persist results on `HIGH_SCORE_LOG` (submission no longer controls deactivation).

### 4.2 Client State
| State | Description |
|--------|--------------|
| Init | Loading protobuf |
| Ready | Normal operation |
| SpeedRunActive | Timer active |
| Prompting | Waiting for name entry |
| Submitted | High score sent |

---

## 5. Typical Message Flows

### 5.1 Startup
1. Client connects
2. Sends CONFIG
3. Server replies with batch, train, mesh data

### 5.2 Normal Interaction
- `Space` → Next dims
- `Enter` → Next batch
- `DIMS` → Accumulate updates
- `SGD_STEP` → Perform step

### 5.3 Speed Run Flow
1. Client → Server: `CONTROL_SPEED_RUN_START` → server resets weights, sets deadline, starts run.
2. Server → Client: TRAIN/VAL/MESH messages include `speed_run_seconds_remaining` while active; client starts local timer for smooth HUD updates.
3. Server: At deadline, schedules a final `VAL_LOSS` evaluation.
4. Server → Client: `CONTROL_VAL_LOSS` with `speed_run_finished=true`; includes the authoritative final validation loss.
5. Client: Shows prompt with the achieved final VAL loss; on valid 4-char name, sends `CONTROL_HIGH_SCORE_LOG` (single submission).
6. Server: Validates name and persists the score and run log.

---

## 6. Input / Output Summary

| Source | Type | Example |
|---------|------|----------|
| **User → Client** | Keyboard | R, Space, Enter, [ ] |
| **Client → Server** | Control messages | CONFIG, DIMS, NEXT, SGD, RUN, LOG |
| **Server → Client** | Control messages | BATCH, TRAIN, VAL, MESH (+timer) |
| **Server → Disk** | SQLite | High scores table with packed log |

---

## 7. Decisions, clarifications, and open items
This section records decisions made and what remains open.

1) Countdown signaling (implemented):
- Included in TRAIN/VAL/MESH while a run is active; client smooths UI locally.

2) Final score reporting and end-of-run (implemented):
- End-of-run is signaled by a `CONTROL_VAL_LOSS` with `speed_run_finished=true` (no special message type). The client only ends on this signal. `CONTROL_FULL_LOSS` was removed.

3) Boundary of run expiry and logging (implemented):
- Strict cutoff—only requests received before the deadline are counted and logged.

4) Weight reset timing (accepted as-is for now):
- Server sets `force_reset_weights` on run start. If generation tracking is needed later, we can add a per-client `weights_generation` to jobs and worker.

5) High score name handling (implemented):
- Server enforces exactly 4 alphanumeric uppercase characters; invalid names are rejected and not recorded. Client UI trims and uppercases.

6) DIMS accumulation and ordering (clarification):
- Per-connection FIFO ordering; `DIMS` applied against the active dims offset at receive time; `NEXT_DIMS` increments the offset.

7) Request index semantics (accepted risk):
- `request_idx` is correlation only; no global dedupe.

8) Multiple runs per session (confirmed):
- Starting a new run resets the in-memory run state/log and starts fresh.

9) Countdown drift and authority (clarification):
- Server is authoritative for end-of-run; client’s 300ms local ticker is UX-only.

10) Security, rotation, and leaderboard retrieval (deferred):
- No auth, rotation policy, or public leaderboard endpoint in scope yet.

---

## 8. Testing Guidelines
- Backend: shorten timer with `HUDES_SPEED_RUN_SECONDS=5`.
- Frontend: unit test HUD markup; E2E verifies end-of-run on final `VAL_LOSS` with `speed_run_finished=true` and high score submission. UI countdown updates locally between server messages.

---

## Appendix: Field Mapping Examples
| Python | JS (protobuf.js) |
|---------|------------------|
| `train_loss_and_preds.train_loss` | `trainLossAndPreds.trainLoss` |
| `speed_run_seconds_remaining` | `speedRunSecondsRemaining` |
| `speed_run_finished` | `speedRunFinished` |
| `dims_and_steps` | `dimsAndSteps[{dim, step}]` |
---

## UI polish and accessibility
- HUD bottom status shows countdown with 1 decimal precision during Speed Run (client-side formatting).
- The “SPEED RUN” control label is styled red, bold, and italic in the controls list for discoverability.
