# Speed Run and Leaderboard Design

## Overview
We are adding a timed “Speed Run” mode to the web app that lets a user train the MNIST CNN as fast as possible via interactive controls. While active, the backend maintains a countdown clock. When time expires, the client is prompted with their score and can submit a 4-character name to a global leaderboard; the server stores the actions log (protobuf replay) for audit and replay.

This document captures: protocol changes, backend changes, frontend changes, storage, scoring, and tests.

## Protocol changes (protobuf)
- New Control.Type values:
  - CONTROL_SPEED_RUN_START: Start a speed run for this client and reset state.
  - CONTROL_HIGH_SCORE_LOG: Submit a high score at the end of the run.
- TrainLossAndPreds: add optional int32 speed_run_seconds_remaining. If a speed run is active for the client, this must be populated in every CONTROL_TRAIN_LOSS_AND_PREDS response.
- New messages:
  - HighScore { required string name = 1; optional double score = 2; optional int32 duration_seconds = 3; optional int32 request_idx = 4; }
    - The server keeps the authoritative action log; the client does not upload it.

Notes:
- We use proto2 syntax so “optional” fields are valid. JS uses protobufjs and will see lowerCamelCase fields automatically.
- We will duplicate hudes.proto under web/public and re-generate hudes_pb2.py after edits.

## Backend server changes
File: `hudes/websocket_server.py`

Constants:
- SPEED_RUN_SECONDS = 60 (configurable for tests; default 60, tests may use 5)

Per-client state additions (extend Client dataclass):
- speed_run_active: bool
- speed_run_end_time: float
- speed_run_log: list[bytes] (ordered, raw serialized Control messages from client)
- best_val_loss_during_run: float | None (for scoring)

Lifecycle:
- On CONTROL_SPEED_RUN_START:
  - Create a fresh Client state for this client_id (reset fields like dims_offset, batch_idx, steps, sgd counters, etc.).
  - Reset the client’s weights server-side by instructing the inference worker to delete client_weights[client_id]. On the next use, weights will reinitialize from mad.saved_weights[float32].clone() automatically (no new client_id and no stored snapshots).
  - Set speed_run_active=True, speed_run_end_time=now+SPEED_RUN_SECONDS, speed_run_log=[], best_val_loss_during_run=None.
  - Force update to push a fresh batch and training results; include countdown in TrainLossAndPreds.

- While speed_run_active is True:
  - Ignore/disable SGD requests (CONTROL_SGD_STEP): do not queue inference_q for sgd; acknowledge no-op or simply ignore.
  - Record all user messages received from this client into speed_run_log (bytes) in arrival order.
  - For each outgoing CONTROL_TRAIN_LOSS_AND_PREDS message, populate speed_run_seconds_remaining = max(0, ceil(end_time - now)). When 0, the server should stop accepting control messages that would affect state (except HIGH_SCORE_LOG).

Post-expiry behavior:
- After `now >= speed_run_end_time`:
  - speed_run_active becomes False. The user can continue interacting normally; training/mesh/val operate as usual.
  - SGD is allowed again (server no longer ignores CONTROL_SGD_STEP after the run is finalized or expired).
  - Do NOT append any further client Control messages to speed_run_log.
  - Freeze best_val_loss_during_run; it will no longer update from post-expiry validation results. The final score is unaffected by post-expiry actions.
  - It is acceptable to continue sending TrainLossAndPreds with speed_run_seconds_remaining=0 for a short grace period; the frontend will already have prompted.

Restart semantics:
- If CONTROL_SPEED_RUN_START is received while a run is active or after one has just completed in this session:
  - Immediately void/clear any previous speed run state and logs (treat as if prior run never existed).
  - Reset weights as above and start a brand-new run with a fresh timer and empty log.

- On CONTROL_HIGH_SCORE_LOG:
  - Accept timing: may be submitted at any time during an active speed run, or after the run has ended, as long as it has not been submitted before for this run.
  - Single-submit rule: allow at most one CONTROL_HIGH_SCORE_LOG per speed run; on receipt, finalize the run (even if timer has not yet expired).
  - Validate: name must be exactly 4 alphanumeric chars (server validates and uppercases).
  - Score selection: score = best_val_loss_during_run (lower is better). Track the best (lowest) validation loss seen during the run.
  - Persist an entry: timestamp, name, score, best_val_loss, duration, request_idx, and the speed_run_log (as BLOB) into a SQLite database via `hudes/high_scores.py`.
  - Clear speed_run_active and log for this client.

Inference loop integration:
- When preparing train/mesh responses in `inference_result_sender`, add the remaining seconds field to TrainLossAndPreds when active.
- When `now >= end_time`, set remaining to 0 and keep responding with remaining=0; the client will prompt and then send HIGH_SCORE_LOG.

Reset semantics:
- “Reset the network to initial settings” means: per-client accumulated weights go back to the saved initial weights; dims_offset, steps, request counters stay consistent but behaviorally new (e.g., batch resets to 0). We can explicitly set batch_idx=0, dims_offset=0, next_step={}, current_step=None, sgd counters=0.

## Storage format (SQLite)
File: `hudes/high_scores.py` manages persistence in a SQLite DB.
- File path: `hudes/high_scores.sqlite3` (override with env var `HIGH_SCORES_PATH` in tests).
- Table: `high_scores`
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - ts TEXT NOT NULL          -- ISO 8601 timestamp
  - name TEXT NOT NULL        -- 4-character uppercase
  - score REAL NOT NULL       -- equals best_val_loss; lower is better
  - best_val_loss REAL NOT NULL
  - duration INTEGER NOT NULL
  - request_idx INTEGER       -- last request index included
  - log BLOB NOT NULL         -- protobuf interaction log (raw bytes; compression optional later)
- Indexes:
  - CREATE INDEX IF NOT EXISTS idx_high_scores_score_ts ON high_scores(score ASC, ts ASC);
- Top 10 query:
  - SELECT name, score, ts FROM high_scores ORDER BY score ASC, ts ASC LIMIT 10;
- Helper methods:
  - init_db(path)
  - insert_high_score(name, score, best_val_loss, duration, request_idx, log_bytes, ts=None)
  - get_top_scores(limit=10)
  - get_all_scores(offset=0, limit=100) (optional)

## Frontend changes
Files: `web/client/HudesClient.js` (+ possibly new UI overlay in `web/client/hud.js` or `View.js`)

Add UI affordances:
- A Speed Run button (or keyboard shortcut) triggers sending CONTROL_SPEED_RUN_START.
- When first CONTROL_TRAIN_LOSS_AND_PREDS message includes speedRunSecondsRemaining, start a countdown timer client-side using it as the authoritative clock. Show timer in HUD.
- When the countdown hits zero, show the prompt. The user may continue to interact; these post-expiry interactions should not impact the recorded high score or the server-side speed run log.
- Disable only the SGD control on the client-side while speed run is active (server enforces ignoring SGD requests). All other interactions (dims updates, next dims, next batch, config) remain enabled.
- When countdown reaches 0, show a modal: Your score is <score>. Enter 4-letter name: [____]. Submit posts CONTROL_HIGH_SCORE_LOG with name and current score. After submit, clear speed run state and show a success toast.

Protocol from client:
- CONTROL_SPEED_RUN_START: empty besides type and requestIdx.
- CONTROL_HIGH_SCORE_LOG: includes 4-char name (server computes authoritative score; client may include its view for UX only).

Scoring rule (initial):
- Use the best validation loss observed during the run, score = best_val_loss (lower is better). Client shows both val loss and score. Server stores both raw best_val_loss and score.

## Tests
Backend (pytest):
- test_speed_run_enable: send CONTROL_SPEED_RUN_START; verify server marks client.speed_run_active and seconds remaining included in next TrainLossAndPreds.
- test_speed_run_resets_state: pre-run do ~10 controls; then start speed run; verify dims_offset/batch_idx reset and client_weights checksum equals initial.
- test_speed_run_interactions_and_high_score: within active run, do 10 interactions (dims, next batch, next dims), then simulate time expiry and send HIGH_SCORE_LOG; verify entry written and log has 10+ messages matching what was sent.
- test_high_score_log_decoding: read the DB file, fetch the BLOB log, parse messages with protobuf, verify they match actions.

Frontend (web mock):
- speedrun.test.mjs: mock WebSocket and protobuf loader; simulate a 5-second run by faking CONTROL_TRAIN_LOSS_AND_PREDS with decreasing remaining seconds; verify the prompt appears and sends HIGH_SCORE_LOG payload with 4-char name.

## Rollout plan
- Implement protobuf and code changes behind minimal UI. Keep DB file in repo .gitignore or tests write to tmp dir. Add env var HIGH_SCORES_PATH override for tests.
- Add unit tests with shortened SPEED_RUN_SECONDS=5 via module-level override or monkeypatch.
