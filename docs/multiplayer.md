# Multiplayer

Open `?lobby=1` on desktop to enter the chat room. Multiplayer is desktop-only: mobile entry displays that requirement and offers solo play without connecting to the lobby or team gameplay. The usual solo entry and `?mode=1d` remain available. Team games use the desktop 3D landscape view; solo mobile play is unchanged.

## Player flow

1. Choose a display name. Create a team or accept a captain's invitation. Only the captain names the team and sends invitations; there is no public join request.
2. The captain starts a team speed run or challenges another captain. Teams contain one to three players and need not have equal sizes.
3. Once everyone confirms readiness and connects to gameplay, the server starts a common countdown. The default preset is three seconds of countdown followed by 120 seconds of play.
4. Everyone sees three 31×31 landscapes, using the same renderer and resolution as solo 3D play. Left/right Shift, clicking a landscape, or right-click cycling selects the enlarged landscape that WASD/scroll controls. Each member can select their assigned landscape plus any unassigned landscapes: a one-player team can use all three, two players share the third, and three players each control their own. Both players may move the shared landscape; the server applies accepted moves in order. Selecting an existing landscape costs no dimension switches. Assignments follow the fixed roster, including disconnected members, and selection survives status updates and gameplay reconnects. Movement changes one shared team model; all teammates see the same losses and landscapes. Opponents have separate models with the same initial settings.
5. New Dimensions casts a vote. A green border marks that player's landscape on every teammate's screen. Players can continue moving while waiting. Unanimity advances directions once and clears every vote. Disconnecting does not reduce the required vote count.
6. At the deadline, the server stops accepting input, completes accepted movement and evaluates final validation loss. Lower loss wins; equal scores tie. A failed evaluation cannot produce a winner. Return to Lobby keeps the team available for another match.

Before starting a speed run or sending a challenge, the captain chooses batch size (1–512) and an optional maximum dimension count (multiples of six, 6–6006). The limit includes the six starting directions: 18 dimensions allows two switches. Blank means unlimited. Challenges display these immutable settings before acceptance; both teams use the same settings. Each team spends its own switch budget only when all its members vote. At zero switches remaining, movement and batch navigation continue until the normal deadline. The gameplay HUD shows the authoritative remaining count, including after reconnect.

Batch navigation and mesh sampling scale are shared within a team. Model, precision, batch size and dimension budget are fixed for the match. Match timing is authoritative on the server; browser clocks are only a display.

In solo play, `;` cycles batch size and Enter selects the next batch. In multiplayer, Enter selects the next shared batch, while batch size stays at the captain’s chosen value (32 by default). Direction refresh advances all six displayed directions; only actual team members participate in the vote.

## Module boundaries

| Module | Owns | Boundary |
| --- | --- | --- |
| `hudes/lobby/` | Identity, chat, invitations, teams, challenges, readiness | Versioned JSON on `/lobby/ws`; injected `GameControl` |
| `hudes/game_control.py` | Immutable roster, preset, admission and status values | Async prepare/start/cancel/status and status subscription; no Torch imports |
| `hudes/game/` | Team sessions, admission, ownership, votes, deadline, inference and scoring | Protobuf on `/ws`; private inference engine |
| `web/lobby/` | Social socket, snapshots and lobby UI | `mountLobby({root, wsUrl, onLaunch, onSolo})` |
| `web/client/` | Game input, rendering, shared game state and results | Launch descriptor plus idempotent `destroy()` |
| `web/app.js` | Screen mounting and return to lobby | Keeps the lobby connection alive during gameplay |

The production server composes both backend modules in one process. Lobby code does not import gameplay rendering, protobuf or Torch. It receives immutable game status and private admission credentials through the control interface, never tensors or mutable inference state. Moving either module to another process would require an adapter for that interface rather than rewriting team rules.

The frontend development proxy forwards both socket paths. A production reverse proxy must also forward WebSocket upgrades for **both `/ws` and `/lobby/ws`** to the backend, preserving the path. The proxy configuration is managed outside this repository; deploying only the web assets cannot enable the new endpoint. Existing `/api` forwarding remains on the HTTP API port.

Gameplay credentials bind one player to one team session and match. The server enforces direction ownership. Input and output carry generation and revision identifiers; inference jobs capture their parameters when accepted. Accepted moves survive a direction change, while obsolete surfaces cannot replace current surfaces. Partial votes do not schedule inference.

The real team engine serializes work on a private worker and keeps a separate weight vector for each team. It does not use the existing solo client's scheduler or weight storage. See [the wire contract](lobby_protocol.md) and [the implementation plan](../multiplayer_plan.md) for lifecycle and ownership details.

## Lifetime and limits

Teams, chat and match history are in memory. They survive matches and temporary connection loss, but not a server restart. Identity resumes through a per-tab browser token. An explicit captain departure disbands an idle team. Busy rosters remain fixed; reconnecting restores the assigned slot. Default idle reconnect grace is five minutes. Preparation, readiness, admission, histories and outgoing queues are bounded; policy values live in `hudes/lobby/models.py`.

Team results do not enter the solo high-score table. Full validation now evaluates every example, including a final partial batch; older solo scores computed with the previous incomplete validation loop are not numerically comparable to the corrected metric. Existing score rows are retained.

## Running and testing

Use the repository's pinned Python and Node versions. Build a CPU test image:

```sh
docker build --target test --build-arg TORCH_BACKEND=cpu \
  --build-arg TORCH_REQUIREMENTS=requirements-cpu.lock -t hudes-test .
```

Run Python tests through the isolated runner, then frontend checks:

```sh
HUDES_TEST_IMAGE=hudes-test scripts/test-python.sh -m pytest -q -p no:cacheprovider /app/tests
cd web
npm ci
npm run test:unit
HUDES_TEST_IMAGE=hudes-test npm run test:protocol
HUDES_TEST_IMAGE=hudes-test npm run test:multiplayer
HUDES_TEST_IMAGE=hudes-test npm run test:e2e:chromium
```

`scripts/test-python.sh` mounts the checkout read-only in Docker and creates a temporary score database. `HUDES_PYTHON` can select a compatible local interpreter instead. Browser fixtures allocate test ports, refuse the live ports 10000/10001, create their own database, and clean up only resources they own. Do not point tests at a production server or score database.

The tests deliberately cover different boundaries:

- Lobby domain and transport tests use fake game control and require no dataset.
- Team session tests inject a clock and stalled engine to exercise ownership, voting, cancellation and deadline races.
- Numerical tests use real PyTorch models and verify combined team movement, opponent isolation and full-dataset scoring.
- Integration tests use actual lobby/game sockets and shared wire fixtures. JavaScript/Python protobuf round trips check the serialization boundary.
- Multiplayer browser tests use actual services with deterministic inference to cover team formation, gameplay, votes and return to the lobby. Legacy browser tests run separately against real inference.

The legacy browser harness runs serially against the real MNIST FFNN model on CPU by default. Set `HUDES_TEST_MODEL=cnn3` to exercise CNN3, and select the appropriate test device/server for its much larger landscape workload. Its default 31×31 grids and batch size of 256 can exceed browser test timeouts on CPU when multiple clients queue inference. This harness choice does not change the production model or game defaults.

A standalone social test server is also available:

```sh
python -m hudes.lobby.server --host 127.0.0.1 --port 18110 --fake-game
```

That explicit fake mode supports social development; it is not an inference server. GPU capacity and deployment behavior need a deployment-specific smoke test before rollout.

## Verification recorded on 2026-10-23

- **162 Python tests passed**, including all 45 existing tests, independent module and socket tests, numerical invariants and production `run_wrapper` MNIST 1v3/3v3 runs.
- **13 multiplayer Chromium scenarios passed**: all nine ordered team-size pairings, repeated cooperative runs, persistent green frames, opponent isolation, independent socket reconnects and mobile denial including `?mobile=0`.
- **25 legacy Chromium scenarios passed** against real MNIST FFNN CPU inference, covering solo 1D/3D, controls, help, speed runs, score insertion and mobile solo play.
- Frontend unit tests, the production Vite build, four bidirectional JavaScript/Python wire fixtures, Ruff and whitespace checks passed.

Python verification used the compatible pinned runtime image with current source mounted read-only, including a `/tests` layout check matching the Docker test target. The Docker test target now copies the authoritative pytest configuration. A fresh full image rebuild and production GPU capacity test were not performed. No production deployment was made. Existing framework deprecation/performance warnings and the large frontend bundle warning remain.

## Live deployment on 2026-10-23

Release `multiplayer-20261023-172538-pdt` was subsequently built from the runtime Docker target and deployed to `https://humandescent.net`. The host `websocket-server.service` systemd drop-in now pins the new image; its existing read-only filesystem, capability restrictions, environment file, data mounts and resource limits were preserved. Direct container replacement is insufficient because systemd recreates the pinned image.

The public nginx configuration now forwards `/lobby/ws` and serves the new versioned web release. Previous assets, proxy configuration, service configuration, image and fresh SQLite backups were retained for rollback.

A complete live 120-second GPU CNN3 3v3 match passed: six player admissions, shared movement, teammate-only green votes, unanimous refresh, finite final validation scores and return to the lobby. Live solo 1D/3D inference and mobile multiplayer denial also passed, with no browser page errors. All 20 existing scores remained byte-for-byte unchanged. This is a functional production smoke check, not a sustained capacity benchmark.

### Solo-style landscape follow-up

Release `three-landscapes-20261023-180116-pdt` now displays three 31×31 landscapes for every team size, matching solo resolution and rendering. Each member's assigned panel stays enlarged; other panels remain read-only. Refresh advances six directions while the voting quorum remains the actual roster size.

Verification included 42 focused backend tests, then 65 contract/lobby/session tests after changing the default resolution; all 13 multiplayer browser scenarios; frontend unit tests including identical solo/team geometry, opacity and transforms for every owner; four bidirectional wire fixtures; formatting and the production build. A complete public 120-second GPU 1v2 match verified all three panels, fixed ownership focus, movement, unanimous voting, finite final scores and return to lobby. Public solo 1D/3D and mobile multiplayer denial passed. Both score databases retained their 20 existing entries. Fresh verified SQLite backups and rollback configurations are under `~/.local/share/human-descent-backups/20261023-180116-PDT-three-landscapes`.

### Custom settings follow-up

Release `match-settings-20261023-181014-pdt` adds captain-selected batch size and an optional dimension budget. The small inference-free `hudes/match_settings.py` module owns validation and budget arithmetic; immutable match presets cross the lobby/game boundary. Public challenge and match snapshots include the agreed settings, while gameplay publishes its authoritative remaining switches over protobuf. No additional socket or shared mutable lobby/game state was introduced.

Verification: 129 focused backend tests passed; the three real-model tests also passed after adding a direct 17-example loss check with a finite budget. All 13 existing multiplayer browser scenarios and two new settings scenarios passed, along with frontend units, six bidirectional wire fixtures, lint and the production build. A complete public 120-second GPU 1v2 match used batch size 64 and a 12-dimension budget, exhausted one team's switch independently, continued moving, finalized finite scores and returned all players to the lobby. Live solo 1D/3D and mobile multiplayer denial passed. Both databases retained all 20 scores; verified backups and rollback files are under `~/.local/share/human-descent-backups/20261023-181014-PDT-match-settings`.

### Solo rendering parity follow-up

Release `solo-render-20261023-181850-pdt` uses the solo starting sampling scale, `10 ** (-0.05 * 12)` (approximately 0.251188643), instead of multiplayer's former 0.1. Both clients initialize the same grid defaults. Team status updates only reconfigure the camera/render targets when the grid layout changes; ownership remains fixed and input remains restricted to the assigned panel.

A Chromium test compares actual canvas pixels for identical input data and each of the three selected panels; solo and multiplayer outputs match exactly. A real-model test compares the actual solo and team loss grids with matching settings. All 58 focused backend tests, frontend units and six focused browser scenarios passed. A complete public GPU 1v2 match confirmed 31×31 meshes, the aligned starting scale, fixed ownership, continued movement after budget exhaustion, finite scoring and return to lobby. Live solo 1D/3D and mobile multiplayer denial passed. Scores remained unchanged; verified backups and rollback files are in `~/.local/share/human-descent-backups/20261023-181850-PDT-solo-render`.

### Compact teammate HUD

Frontend release `compact-hud-20261023-182400-pdt` moves the teammate display into a slim strip eight pixels beneath the loss-plot container, confined to the left two plot columns. Compact name chips show landscape colors, ownership, green request borders and offline states; the strip retains readiness and remaining switches. Loss, time and batch values remain in the existing bottom HUD. Long names truncate visually and retain full labels/tooltips. Match results use a separate centered card after play ends.

Frontend units and three browser scenarios passed, including a 3v3 geometry check confirming the HUD sits below the plots, left of the prediction column and within the compact height limit. Screenshots were visually reviewed; the public HTML/CSS/JavaScript bytes match the tested build. The frontend deployment required no backend restart. The previous nginx configuration is retained in `~/.local/share/human-descent-backups/compact-hud-20261023-182400-pdt`.

### Landscape selection for smaller teams

Release `panel-selection-20261023-191444-pdt` enables the usual Shift, click and right-click landscape selection for one- and two-player teams. One player can control all three panels; two players can each control their assigned panel and the unassigned third. Three-player ownership stays fixed. WASD and scroll move the selected landscape. Selection survives status updates, direction refreshes and gameplay reconnects, costs no dimension budget, and leaves vote indicators attached to each member's assigned panel. Disconnected members retain ownership. The server independently validates every movement against the fixed roster and combines shared-panel moves through its existing inference queue. Pointer-capable browsers now handle pointer events without duplicate compatibility mouse events, preventing double selection/cycling.

Verification passed: all 189 Python tests, frontend units, six bidirectional protocol fixtures, production builds, Ruff and whitespace checks. Browser coverage includes all nine team-size pairings, selection with exhausted budgets, actual mouse clicks/right clicks, shared movement, ownership restrictions, refreshes and reconnects. The final focused multiplayer run passed all 12 scenarios; solo/mobile checks also passed after the mouse correction. Browser runs use separate artifact directories when executed concurrently.

The managed backend and versioned nginx frontend were deployed to the public site. A full live 120-second GPU CNN3 1v2 match verified selection, combined movement, teammate protection, independent voting, budget exhaustion, reconnect focus, finite final scores and return to lobby. Live solo 1D/3D inference and mobile multiplayer denial passed without browser page errors. All eight public HTML/JS/CSS files match the staged build, and all 21 production scores and 20 workspace scores remain unchanged. Existing service hardening was preserved. Verified database backups, previous configurations/image references, rollback scripts, screenshots and check logs are under `~/.local/share/human-descent-backups/panel-selection-20261023-191444-pdt`. Cutover checks wait for both the intended backend image and nginx reload to become visible before declaring success.

### Mouse listener cleanup

Frontend release `mouse-cleanup-20261023-202304-pdt` fixes `KeyboardClientGL.destroy()` to call the mouse controller's `dispose()` and release its reference. The lifecycle regression now installs real mouse controls and leaves mid-drag, covering both pointer and mouse-only input across repeated mounts and idempotent teardown. It failed before the fix because movement still reached the disposed view, and passes afterward with no canvas/window handlers retained.

All frontend units and the repeated three-player match browser scenario passed. A public-browser check confirmed all three installed mouse window listeners are removed and repeated destruction is safe. This was a frontend-only deployment; the backend was not restarted. The previous nginx configuration is retained under `~/.local/share/human-descent-backups/mouse-cleanup-20261023-202304-pdt`.
