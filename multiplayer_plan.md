# Multiplayer plan

## Scope and agreed behavior

The first multiplayer mode is cooperative play on assigned landscapes for a team of one to three players. Players share **one model per team**, including its weights, batch/index, batch size, precision, settings, loss, and direction set. Each player controls only their assigned landscape (two parameter directions). Every player receives the same shared inference and landscape results and can see the team's activity. Opposing teams have separate models.

Keep the implementation small: add shared-session membership, routing, broadcasts, and server-side control ownership checks, while reusing the existing inference math and landscape rendering. This is not a separate model per player or a redesign of the inference architecture.

### New Dimensions is a unanimous ready vote

- A player's New Dimensions action requests the next directions for the current direction generation. It does not immediately refresh the team.
- That player's assigned panel gets a persistent **green box border**, visible to everyone. This indicates readiness independently of temporary movement feedback.
- Each player can request once per generation. Repeated requests are idempotent and do not add votes.
- Players can continue moving while waiting for teammates.
- Once every member has requested (1/1, 2/2 or 3/3), the server advances the shared directions exactly once, preserves model weights and all accepted moves, and clears all ready flags and borders.
- Solo New Dimensions retains its immediate behavior.

## Current implementation touchpoints

The backend is Python/PyTorch with WebSocket/protobuf communication; the frontend is Vite/Three.js.

- `hudes/websocket_server.py` currently holds per-client weights and `Client` state, and `CONTROL_NEXT_DIMS` immediately advances `dims_offset`.
- `web/client/HudesClient.js` currently sends New Dimensions and immediately increments local `dimsOffset` and resets tracking.
- `web/client/views/LandscapeView.js` rebuilds 3D meshes in `updateMeshGrids`; current 1D glows are transient and cannot serve as persistent team readiness state.

## Proposed implementation

These are concrete implementation proposals supporting the agreed behavior, rather than completed changes.

### Shared session and ownership

Introduce a small gameplay `TeamSession` holding shared model/settings state, direction generation, member-to-landscape assignments, and per-member requested flags. Admit authorized players through individual gameplay credentials supplied after lobby match preparation. Social team membership is invite-only; gameplay admission must not become a public team-join operation. Keep credentials individual.

Joining players adopt authoritative session settings instead of overwriting them with browser defaults, which can differ. Schedule shared pending input/inference once and broadcast the resulting shared state to members. Enforce landscape ownership on the server. Keep the session alive while team members remain.

Send a full shared-state snapshot on join/reconnect, including generation, assignments, and ready flags.

### Vote and generation handling

Route team `CONTROL_NEXT_DIMS` through the ready-vote handler; keep the existing immediate path for solo clients. A partial vote broadcasts lightweight readiness state and does not run inference.

Make the final vote's state transition atomic before awaiting network sends: advance the generation/directions once and clear all flags. Tag votes, moves, and results with their direction generation. Capture the generation when work is scheduled, so stale moves, votes, and results cannot be interpreted in the new direction set. Preserve accepted weight changes across the refresh.

Advance by the shared direction count, not merely the requesting player's two directions. The confirmed layout now preserves three visible landscapes for every team size, so a refresh advances all six displayed directions. Players still control only their assigned pair; unassigned landscapes are read-only, and votes are required only from actual team members.

### Frontend state and display

In a team, `getNextDims` sends a vote only. Update local direction counters and reset tracking only when the server announces an authoritative generation change.

Ignore Space autorepeat until key release. On a generation change, stop held movement input until release and show an updating state until the matching landscape surface arrives.

Store readiness independently of meshes. Render a persistent 3D green frame separately from transient 1D movement glows, and reapply it after mesh rebuilds. Add a view-router setter for readiness state and show a count such as `1/3`. Multiplayer is desktop-only; every desktop teammate sees the assigned landscapes and readiness. Mobile continues to support solo play only.

### Disconnects: tentative policy

Retain a member's slot and ready vote during a brief disconnect so a network interruption does not automatically trigger a refresh. Restore the snapshot on reconnect. Explicit leave updates membership; the precise policy for recomputing unanimity after a leave, along with disconnect grace duration and expiry behavior, is still open.

## Relevant files

- `hudes/websocket_server.py`: session state, control routing, shared input/inference, broadcasts.
- `hudes/hudes.proto`: session, generation, assignment, and readiness messages as needed.
- `hudes/control_security.py`: server-side ownership checks.
- `web/client/HudesClient.js` and `ClientState.js`: authoritative session/generation/readiness state.
- `web/client/KeyboardClient.js` and `KeyboardClientGL.js`: vote and held-key behavior.
- `web/client/View.js` and `views/LandscapeView.js`: readiness routing and persistent borders.
- HUD/help-tour code and tests: team readiness instructions and coverage.

## Verification

- First vote changes readiness without scheduling inference.
- Repeated votes are idempotent.
- The last distinct member's vote advances directions exactly once and clears readiness.
- Refresh preserves weights and accepted moves.
- Ready borders survive movement, inference updates, and mesh rebuilds.
- Stale-generation votes, moves, and results are rejected or ignored.
- Reconnect restores member assignment and readiness.
- Solo behavior remains unchanged.
- Exercise two- and three-browser desktop teams. Verify mobile multiplayer entry is blocked before connecting and solo mobile remains functional.

## Secondary mode and future discussion

A possible alternative gives each player independent random subspaces on the same shared weights, settings, and loss, potentially six directions per player. If each player gets only one 2D panel, the mathematical behavior is the same as assigned-landscape cooperation; the main difference is what each player sees. This is a secondary option, not the immediate implementation.

The lobby, invitation, challenge, and service-separation requirements below supersede the earlier open lobby discussion. The confirmed preference for the first team-versus-team format is **lowest loss after a shared countdown**. Exact duration and final-versus-best loss scoring remain proposed defaults rather than explicitly confirmed rules.

Before future ranked team modes, resolve the previously identified scoring prerequisites: completion/update lifecycle problems and the validation-batch mismatch where a default of 1024 is capped at 512 and evaluates 5,120 of 10,000 examples. These should not widen the scope of basic cooperative play.

## Lobby and independent modules: confirmed requirements

- Teams exist in a shared chat room and stay together between runs.
- Multiplayer is desktop-only. Mobile entry must explain the requirement and offer solo play without opening multiplayer sockets; do not add mobile team controls.
- The creator is captain. Only the captain chooses the team name and invites players.
- Invitations can be accepted or declined. There is no public join or request-to-join operation.
- Teams have one to three players. A captain can start a team speed run or challenge another team's captain.
- Challenges allow all size combinations: 1v1, 1v2, 1v3, 2v2, 2v3, and 3v3. No automatic handicap is specified.
- Lobby/chat traffic uses a separate WebSocket and client from gameplay. The existing gameplay socket remains focused on gameplay.
- Lobby state, gameplay state, credentials, and lifecycle ownership must be explicit. No mutable game objects or tensors cross into lobby modules.

## First implementation scope

Ship chat/presence, captain-only teams and invitations, challenges, ready checks, team speed runs, unequal-team matches, assigned-landscape co-op, unanimous direction voting, reconnect snapshots, and returning to chat with teams intact. Implement readiness as a proposed launch safeguard, distinct from in-game direction votes.

Defer independent six-direction subspaces per player, persistent accounts/team history across server restarts, public ranked team leaderboards, matchmaking, spectators, direct messages, voice chat, and multiple inference servers. Existing solo gameplay remains supported. Display match results without inserting them into the existing solo leaderboard.

Proposed bounded defaults for implementation, subject to product adjustment:

- One social team per player and one reserved/active match per team.
- Every team sees three landscapes, matching solo 3D. Each member controls one assigned landscape, which remains the only enlarged panel in their view. Smaller teams have read-only unassigned panels. Six directions are displayed, but each player is authorized to move only their assigned two.
- Start with a 120-second preset and final validation loss from a frozen model; keep duration and scoring policy explicit in the preset. Do not inherit the existing best-score implementation accidentally.
- Match presets must specify the actual initialized model, batcher, and direction-basis seeds. The current `Config.seed` field is stored without configuring those objects and is not evidence of reproducible match initialization.
- Freeze match rosters before countdown. Roster changes before that invalidate readiness/challenges. Temporary disconnection preserves membership, assignment, and vote; it never counts as a new vote or removes a required voter.
- Use a bounded preparation/readiness timeout; cancel preparation and release both teams on failure. During play the game clock continues through a temporary disconnect. Report inference failures as failed matches, not opponent wins.
- Captain role persists during a reconnect grace period. Explicit captain departure while idle disbands the team; automatic succession is deferred. Do not mutate a running match's roster through a social leave operation.
- Teams and bounded recent chat history persist in server memory across runs, not across server restarts.
- Preserve the existing solo entry URL and add an explicit lobby entry initially. Lobby membership can stay connected while gameplay is mounted. Team landscape/vote visibility means teammates; opponent landscape streaming is outside this first scope.
- A tied score remains a tie. Failed/nonfinite evaluation cannot produce a valid winning score.

## Architecture and contract gate

Use independent modules in the same repository, initially composed in one server process. Separate standalone lobby startup should work with a fake game-control implementation and without importing Torch or loading MNIST. A single process does not promise process-crash isolation; separate deployment is later work.

Proposed locations:

| Boundary | Files | Ownership |
| --- | --- | --- |
| Game-control contract | `hudes/game_control.py` | Coordinator |
| Lobby protocol and interface examples | `docs/lobby_protocol.md`, `tests/fixtures/multiplayer/` | Coordinator |
| Lobby service/transport | `hudes/lobby/` | Lobby backend agent |
| Gameplay sessions/control adapter | `hudes/game/`, existing gameplay backend files | Gameplay backend agent |
| Lobby client/state/UI | `web/lobby/` | Lobby frontend agent |
| Gameplay client/views | Existing `web/client/` files | Gameplay frontend agent |
| App composition and integrated tests | Existing app entry, build/test configuration, integration suites | Integration/verification agent |

Recommend a versioned JSON protocol at `/lobby/ws` and retain protobuf at `/ws`. Lobby JSON must be validated and bounded. Do not multiplex lobby messages into gameplay `Control`. The current server does not actually route by WebSocket path, so explicit dispatch is required. Preserve legacy direct `process_client` tests/native clients when introducing routing.

Freeze these interfaces and representative fixtures before parallel implementation:

1. **Lobby wire contract:** command version/type/request ID/payload; request-correlated response; stable errors; ordered state events; full reconnect snapshot. Actor identity is derived from the connection. Public events exclude all credentials. Commands cover identity/resume, chat, create/rename/leave team, invite/accept/decline, start speed run, challenge/accept/decline/cancel, and readiness. No join-request command exists.
2. **Game-control contract:** immutable roster/preset DTOs, idempotent `prepare_match`, `start_match`, `cancel_match`, `get_match_status`, plus prepared/started/finished/failed/cancelled events. Preparation returns match/session IDs and individual admission credentials through private delivery. IDs identify the correct player/team/match. Only the game service owns the authoritative start, deadline, accepted-input cutoff, and final score. The lobby owns social readiness and requests start after the game reports required connections prepared; it only mirrors game lifecycle afterward.
3. **Gameplay protocol additions:** match/session identity, server-authorized panel assignment, direction generation on moves/votes/results, model revision, private admission/resume, and shared vote state. Capture immutable job identity when scheduling inference; never stamp late results with the current mutable generation.
4. **Browser launch/lifecycle contract:** a launch descriptor contains only that player's endpoint, credential, match/session ID, assignment and authoritative settings. Gameplay exposes a mount/start and dispose/stop lifecycle. Lobby and game use separate connections. App composition is the only frontend bridge between them.

Repeated start/accept/cancel and repeated lifecycle events are idempotent. Reserve both teams before awaiting preparation; release them on cancellation/failure. Ignore obsolete preparation responses and prevent duplicate matches. A status snapshot is the recovery path when events are missed. Keep the initial adapter in-process; an internal HTTP service or generic message bus is unnecessary.

## Parallel SOL extra-high implementation plan

Use `gpt-6.1-sol` with reasoning effort `xhigh` for each implementation agent, with fresh bounded context (`fork_turns="none"`) and explicit file ownership. Five implementation agents plus the coordinator are sufficient. More agents should not concurrently edit the same runtime or renderer.

### Wave 0: contracts and reproducible baseline

The coordinator freezes the four contracts above, schema examples, policy defaults, ownership, and acceptance criteria. The integration/verification agent starts runtime discovery, isolated fixtures, and fake game-control support in parallel. It may scaffold tests against the agreed contract but does not invent a competing wire format.

The existing local Python environment has an incompatible protobuf runtime; local Node/modules also differ from declared pins. Establish a reproducible test runtime before treating failures as implementation regressions. Use the repository's CPU-lock Docker test target for the canonical backend baseline; an existing compatible local image can be used after checking its versions. Use the declared Node version and `npm ci` in an isolated worktree/runtime. Do not bypass protobuf checks, edit generated headers to suppress errors, or opportunistically update dependency locks.

The planning audit observed Python protobuf 6.33.0 against generated 7.35.1, Node 22.12.0 against required 26.12.1, and stale installed frontend packages. The existing `human-descent-security:xrender-test` image was checked with a disposable read-only container: its protobuf 7.36.2 successfully imports this checkout's generated schema. This is an environment check, not a passing application baseline or a clean rebuild. The default Playwright `dev-data/test-venv/bin/python` is absent. Schema generation also requires a compatible compiler, independently of the runtime.

### Wave 1: four component agents and one integration agent

| Agent | Exclusive write scope | Deliverable and local proof |
| --- | --- | --- |
| A: Lobby backend | `hudes/lobby/**`, `tests/test_lobby_*.py` | Torch-free chat/team/invite/challenge service and separate JSON WS, fake-clock and fake-game tests; captain permissions, invitation races, all team size pairings |
| B: Gameplay backend | `hudes/game/**`, `hudes/websocket_server.py`, `hudes/control_security.py`, `hudes/model_data_and_subspace.py`, `hudes/hudes.proto`, generated protobuf outputs, `tests/test_team_game_*.py`, `tests/test_game_scoring_*.py` | Shared model/team routing, admission, generation-safe unanimous votes, snapshots, preparation/start/freeze/results; real-model scoring correctness; explicit socket dispatch/composition |
| C: Lobby frontend | `web/lobby/**`, `web/tests/lobby_*.test.mjs` | Independent lobby store/client/UI, team cards, captain invitations/challenges, ready flow and result display against fixed wire fixtures; no gameplay client creation |
| D: Gameplay frontend | `web/client/**` including HUD/help and views, `web/tests/team_game_*.test.mjs` | Desktop-only launch descriptor support, owned controls, persistent green frames, authoritative state, all-player readiness, input gating and real dispose lifecycle; legacy solo behavior |
| E: Integration and verification | App entry/HTML/shared `web/style.css`, `web/vite.config.js`, package test scripts as needed, Playwright config/fixtures/new integration specs, `.github/workflows/ci.yml`, pytest test configuration if needed, `tests/test_multiplayer_*.py`, `tests/test_game_control_contract.py`, test-runtime helpers | Fake-backed real-server harness, path/protocol contract tests, browser composition, complete user-flow tests, independent socket failures, CI and real-engine smoke |

Agent B alone owns `websocket_server.py` even when wiring the lobby. Agent E specifies routing expectations through tests and messages, not overlapping patches. Agent D alone owns `HudesClient` and `LandscapeView`; Agent C works through the launch contract. Agent E alone owns shared build/test configuration. The coordinator alone owns this plan and contract documents/definitions until an explicit handoff.

The existing `npm run dev` and build prehooks regenerate protobuf outputs. Agent B generates those outputs in the pinned compiler environment at a coordinated checkpoint. Other agents use generation-free Vite/test commands while editing in parallel; the full normal build runs after schema ownership is released. `web/public/hudes.proto` already points to the canonical schema; do not fork an independent browser copy.

Each component agent writes its own meaningful unit tests. Agent E owns cross-component tests and can proceed early against fakes. Do not make one test agent responsible for discovering every component's requirements after coding is complete.

All agents in the current environment share a workspace. Use the exclusive scopes above; never assume automatic branch/worktree isolation. Separate worktrees are an optional deliberate setup, not the default. Preserve all pre-existing untracked user files, including scripts, logs, datasets, and tutorial tests. Do not commit, publish, deploy, or alter unrelated assets as part of a component task.

### Wave 2: integration and review

Compose lobby and game only after component contract tests pass. Integrate the browser through the launch/dispose interface. Agent E runs the end-to-end matrix, while component owners fix failures in their own files. The coordinator checks dependency direction, scope, contract consistency, and the final diff. Reuse completed agents for bounded cross-review; do not start additional concurrent writers on owned files.

The critical path is shared gameplay correctness, browser lifecycle, then real-engine integration. Parallelism mainly helps the independent lobby and UI work; it does not remove these dependencies. Completion is determined by acceptance gates, not the number of agents or passing fake-only demos.

## Test layers and release gates

| Layer | Backend | Frontend/integration |
| --- | --- | --- |
| Pure behavior | Team permissions, invite capacity/races, expiry with injected clock, challenge reservations, ready state, vote unanimity | Lobby reducers/state, launch descriptor validation, readiness state, input permission logic |
| Protocol | Versioned JSON/error/snapshot fixtures; JS/Python gameplay encoding; path isolation; private admission | Real clients decode canonical fixtures; unexpected/stale events handled |
| Real server + fake game | Actual separate WS endpoints and lobby lifecycle with deterministic fake preparation/failure/results | Multi-context Playwright: invitations, all size combinations, countdown, chat, rematch/return, reconnect |
| Real game adapter | Shared weights exactly once, opponent isolation, read-only surface refresh, generation ordering, fixed cutoff/frozen score, full validation coverage | Representative real-inference team speed run, unequal-team match, and unanimous border/refresh flow |
| Independence/regression | Lobby imports/start/tests do not require Torch/MNIST; lobby traffic never schedules inference; solo/native paths remain valid | Lobby disconnect leaves game functioning; game disconnect leaves chat functioning; repeated entry/exit leaves one set of listeners/connections |

Specific acceptance gates:

- No public join UI or server command; forged captain/recipient/team actions fail.
- Teams cannot exceed three members, join two teams, or reserve two matches through concurrent requests.
- Support 1v1/1v2/1v3/2v2/2v3/3v3 and one-team speed runs without equal-size assumptions. Exercise reversed challenger/recipient sizes too (all nine ordered pairings).
- First/repeated direction votes do not run inference; final distinct vote advances exactly once and clears all borders. 1/1 works in the team mode.
- Joining/reconnecting adopts team settings; device defaults cannot change the team's batch size or precision.
- Old-generation input/results cannot affect the new set; previously accepted movements survive a refresh.
- Keyboard Space in chat never votes; holding Space cannot vote automatically in the next generation; movement key holds reset safely on refresh.
- Readiness borders survive mesh rebuilds, camera/selection changes and ordinary movement on desktop. Mobile cannot enter multiplayer or construct team gameplay; existing solo mobile controls remain available.
- Preparation fails cleanly if one team cannot prepare. No incomplete match starts; no duplicate starts or post-terminal state regressions.
- Match scoring applies the defined pre-deadline inputs once, freezes both models, waits for final evaluation, and compares the same metric/version. Scores are finite; ties/failures have explicit outcomes.
- Correct the validation batch-count/cap mismatch with a ragged-dataset coverage test before accepting competitive scores. This changes validation values; record the evaluation version and do not claim comparability with historical solo scores. Keep team results out of the existing solo leaderboard.
- Repeated start/return cycles dispose listeners, timers, render loops, subscriptions and GPU resources; the existing GL double-input initialization must be resolved within this lifecycle work.
- Isolated test ports and per-run temporary databases; no production listener termination or duration-based deletion from arbitrary score databases. Parallel tests use unique IDs/fixtures and serialize shared real-engine resources when necessary.

Run pure/fake-backed tests on each change. Add the full frontend unit suite, build, protocol tests and fake-backed Playwright suite to CI. Keep a smaller real CPU inference job and targeted GPU responsiveness checks. Fake game tests are not evidence that numerical inference or GPU scheduling is correct.

## Implementation agent handoff template

Each launch brief must include: the relevant confirmed requirements and labelled defaults; frozen contract revision and fixture paths; exclusive files; dependencies that must be ready; commands for unit/contract tests in the agreed runtime; cross-owner change-request procedure; and acceptance gates. Require a completion report with changed files, tests actually run and outcomes, unverified behavior, and any requested contract changes. Report blocked prerequisites promptly; do not silently change another agent's protocol or files.

Status: implemented locally by five parallel SOL extra-high agents, with coordinator review and integration testing. The user subsequently restricted multiplayer to desktop only; this supersedes the earlier mobile exploration. Acceptance passes 162 Python tests (including the legacy suite and real MNIST 1v3/3v3 runs), all frontend units, the production web build, cross-language wire checks, 13 multiplayer browser scenarios and 25 legacy browser scenarios. Desktop readiness frames were visually inspected and checked geometrically from every teammate's view. Mobile multiplayer is rejected before connecting, while solo mobile regressions pass. See `docs/multiplayer.md` for the module boundaries, operating requirements, test commands and remaining deployment validation.

## Match configuration follow-up

Captains configure batch size and an optional total dimension budget before a speed run or challenge. Both opponents agree to the same immutable settings; each team advances independently by unanimous vote. All three landscapes always remain visible. The total includes six starting directions and increases by six per switch. Show switches remaining in the gameplay HUD; zero locks direction changes but permits continued optimization until the deadline. Batch size defaults to 32 and dimension budget defaults to unlimited.
