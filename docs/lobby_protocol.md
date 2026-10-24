# Multiplayer contract v1

Implementation contract for the independent lobby and gameplay modules. This document and `hudes/game_control.py` are coordinator-owned. Request changes explicitly instead of independently renaming fields.

## Boundaries

- `/lobby/ws`: JSON social protocol. Lobby can run with an injected GameControl and without importing Torch, model data, gameplay server, or protobuf.
- `/ws`: existing protobuf gameplay protocol, extended additively for team admission/state. Gameplay backend owns those generated definitions and communicates their exact fields to gameplay frontend and integration owners.
- `hudes.game_control.GameControl`: immutable Python DTOs and async methods connecting the two modules. Same-process composition initially; no internal HTTP server needed.
- Lobby and game use separate sockets and individual credentials. A lobby disconnect does not stop the game. A game disconnect does not close chat.
- The supported multiplayer browser client is desktop-only. Mobile entry must stop before mounting either multiplayer client, show the desktop requirement, and offer solo play. This presentation policy does not add device identity to the wire protocol.

## JSON envelope

Request: `{"v":1,"type":"team.create","requestId":"unique-id","payload":{"name":"Local Minima"}}`.

Response: `{"v":1,"type":"response","requestId":"unique-id","ok":true,"payload":{}}`.

Rejected response: same envelope with `ok:false` and `error:{code,message}`. Use stable codes such as `invalid_request`, `unauthorized`, `not_captain`, `not_found`, `team_full`, `already_in_team`, `team_busy`, `expired`, `conflict`, `not_ready`, `rate_limited`, `game_unavailable`.

Event: `{"v":1,"type":"snapshot","seq":42,"payload":{...}}`. Each connection receives ordered event sequence numbers. Clients adopt full snapshots; they never infer authoritative membership from a local click. Retried commands with the same request ID must not repeat side effects; cache a bounded set per identity. Reject reuse with different content.

Hello/resume: `hello` payload `{name,resumeToken?}`. Success payload `{playerId,resumeToken}`. Server issues identity/token; name is display only. Follow with a private snapshot. An invalid supplied resume token must produce an explicit rejection, not impersonate another identity. Client may then choose a new hello. A resumed identity replaces its old lobby connection safely.

Commands and payloads:

| Command | Payload |
| --- | --- |
| `snapshot` | `{}` |
| `chat.send` | `{text}` |
| `team.create` / `team.rename` | `{name}` |
| `team.invite` | `{playerId}` |
| `invite.accept` / `invite.decline` | `{inviteId}` |
| `team.leave` | `{}` |
| `team.start_speedrun` | `{}` |
| `challenge.create` | `{teamId}` (target) |
| `challenge.accept` / `.decline` / `.cancel` | `{challengeId}` |
| `match.ready` | `{matchId,ready:true/false}` |
| `match.cancel` | `{matchId}` |

No public team join or join-request exists. Only captains create invitations, rename, start runs or challenge; only invite recipients accept/decline their invitations; only the challenged captain accepts/declines a challenge. One team/player, maximum three members/team; all ordered 1/2/3 versus 1/2/3 combinations allowed.

## Snapshot shape

```
{
  you: {id, name, teamId},
  players: [{id, name, connected, teamId}],
  teams: [{id, name, captainId, memberIds, revision, status}],
  invitations: [{id, teamId, inviterId, recipientId, status, expiresAt}],
  challenges: [{id, challengerTeamId, targetTeamId, status, expiresAt}],
  matches: [{id, teamIds, phase, readyPlayerIds, startAt, deadline, results, reason}],
  chat: [{id, playerId, name, text, sentAt}]
}
```

Optional/unavailable values use null. Times on the wire are Unix seconds. Team status is `available` or `busy`; match phase includes lobby `preparing`/`ready` plus GameControl phases. Results use `{teamId,loss}`. Invitations in a snapshot are restricted to the sender/captain and recipient. Public snapshots never include resume/admission secrets. Match history and chat are bounded. Only the player's own match needs readiness detail; public team availability is sufficient for others.

Use safe text rendering for names/chat. Bound messages/history/queues and validate types. Network sends happen after atomic state transitions, outside state locks. Expiry, capacity and reservations are checked on the server, including acceptance races.

## Match lifecycle

Default server-owned preset: duration120s, countdown3s, batch32, 31x31 grids matching solo 3D, step≈0.251188643 (the solo starting scale), float32, seed0, final_validation_loss_v2. Test server may inject shorter durations and smaller grids. Browser cannot override the ranked/scoring configuration through arbitrary commands. Actual model/batcher/basis must match the preset seed; do not rely on the existing unused Config.seed field.

Starting a speed run reserves one team; accepting a challenge reserves both teams atomically. Build immutable MatchSpec with panel indices0..team_size-1, prepare game sessions, and place match in lobby ready phase. A prepare failure releases reservations and cancels partial preparation. Rosters remain fixed while reserved.

Every participant calls match.ready. Once all are ready, privately deliver `game.launch` to each member and wait for GameControl status to report all fixed players connected. Then call start_match once. GameControl starts the common countdown and owns the authoritative monotonic cutoff. A bounded preparation/readiness/admission timeout cancels incomplete matches. A status snapshot recovers missed lifecycle events. Callback implementation must avoid re-entrant deadlocks: do not hold lobby locks while awaiting GameControl operations or network sends.

`game.launch` payload (only to its player):

```
{matchId, teamId, playerId, sessionId, panelIndex, token, wsUrl:"/ws"}
```

On reconnect, resend an own launch descriptor if the match has been launched and is still live. App must deduplicate by sessionId. Gameplay admission supplies the authoritative settings/roster/votes before input is enabled. Never call the legacy solo Config path to enter a team session.

At finished/cancelled/failed, release social team reservations and preserve the teams. Match result stays available in snapshots so return to lobby works after missed events. Individual game sockets report terminal state too, allowing play to finish without lobby connectivity. Completed results cannot be modified by late jobs. Team scores never enter the solo leaderboard.

## Shared gameplay navigation

Team mode is the 3D assigned-landscape interface; a solo `mode=1d` URL does not change a launched team match into the 1D client. Legacy solo 1D remains available.

Team batch size, precision, model, and scoring preset are fixed during a competitive run. Useful exploration controls remain: `CONTROL_NEXT_BATCH` advances the shared batch index, and `CONTROL_CONFIG` with the new optional `team_mesh_step_size` field requests a shared sampling/movement scale. These team commands do not contain legacy Config, whose required fields would overwrite unrelated settings. Both are generation-tagged and request-indexed, validated and deduplicated by the server. The accepted scale/batch is broadcast to all teammates; votes are preserved. Additive `TeamState.batch_idx`/`Control.team_batch_idx` metadata and immutable job tags identify the shared batch. Old surfaces must not overwrite a newer configuration's view.

## Browser module APIs

Lobby frontend exports `mountLobby({root, wsUrl, onLaunch, onSolo})` from `web/lobby/index.js`, returning `{destroy, client}`. `destroy()` is idempotent. Lobby modules do not import game modules or Three.js. Mount can remain connected while its root is hidden during gameplay. The client exposes subscription/state for integration tests without browser-global coupling.

Gameplay frontend preserves existing KeyboardClient constructors and accepts `options.launchDescriptor` and `options.onReturnToLobby`. Add idempotent `destroy()` on the client/view chain. Composition provides existing game DOM only when mounting gameplay, and can keep a separate lobby host connected. Multiplayer result UI includes a Return to Lobby action and invokes onReturnToLobby. Legacy solo entry and controls remain available.

App uses explicit `?lobby=1` entry initially and preserves existing solo URLs. Lobby entry is discoverable from solo via a link/button. Chat input/focus must stop held movement and never invoke gameplay shortcuts. Destroy disables reconnect, clears timers/input repeats/subscriptions, cancels render loops, disposes charts/graphics, removes owned DOM/listeners, and prevents late async callbacks from resurrecting UI.

## Defaults and evidence

Teams persist across runs/reconnects in memory, not across server restarts. Explicit captain leave while idle disbands; running rosters are immutable. Temporary disconnect does not change unanimous vote membership or panel permissions; deadlines continue. Every team receives three landscape meshes and six displayed directions. Each member may control their assigned panel or any panel unassigned in the fixed roster. One player can use all three; two players share the unassigned third; three players each retain their own. Selection is local UI state, costs no budget, and does not change roster assignments or vote indicators. Each movement command contains at most two steps from one allowed panel; shared-panel moves enter the same ordered inference queue. Every direction refresh advances by six. Legacy solo is unchanged. New-dimension request flags stay true while moving; unanimity1/1,2/2,3/3 refreshes once and clears flags. Opposing teams vote independently. Ties remain ties; engine failures are failed matches, not wins. Exact numerical correctness, accepted-move preservation, deadline freezing and validation coverage require real-engine tests; fakes establish only lifecycle/interface behavior.

## Custom match settings

`team.start_speedrun` and `challenge.create` accept optional `settings: {batchSize: 64, maxDimensions: 18}`. Batch size is an integer 1–512. `maxDimensions` is null (unlimited) or a multiple of six from 6 to 6006, including the starting six. Omitted fields inherit server defaults. No other preset fields are accepted. Only captains can create runs or challenges. A challenge captures an immutable preset, publishes its settings, and acceptance uses that exact preset for both teams. To change an outstanding challenge's settings, cancel it and send another. Match snapshots and results retain `settings`.

The optional protobuf `TeamState.dimension_switches_remaining` is absent for unlimited matches, explicitly zero for exhausted budgets. For a finite limit, remaining = max(0, maxDimensions / 6 - 1 - generation). Partial/repeated votes do not spend budget; unanimous votes spend one. The backend rejects new votes when exhausted while accepting movement and other allowed controls. Each opposing team has an independent counter under the same limit. Reconnect receives the authoritative counter.
