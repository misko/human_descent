import { createLobbyClient } from './client.js';
import { deriveLobby } from './state.js';
import { lobbyStyles } from './styles.js';
import { normalizeSettings, settingsLabel } from './matchSettings.js';

export { LobbyClient, LobbyError, createLobbyClient } from './client.js';
export { deriveLobby } from './state.js';

let mountCounter = 0;

// Names, chat and server messages are always text nodes, including attributes
// used to address players. No server content is interpreted as markup.
function element(doc, tag, attrs = {}, ...children) {
  const node = doc.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'disabled' || key === 'hidden') node[key] = !!value;
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child != null) node.append(typeof child === 'string' ? doc.createTextNode(child) : child);
  }
  return node;
}

export function mountLobby({ root, wsUrl = '/lobby/ws', onLaunch = () => {}, onSolo = () => {} }) {
  if (!root?.ownerDocument) throw new Error('mountLobby needs a root element.');
  const doc = root.ownerDocument;
  const mountId = `lobby-${++mountCounter}`;
  const e = (tag, attrs, ...children) => element(doc, tag, attrs, ...children);
  const client = createLobbyClient({ wsUrl });
  let destroyed = false;
  let currentView = deriveLobby(null);
  const drafts = new Map();
  const host = e('section', { class: 'hd-lobby', 'data-testid': 'lobby', 'aria-label': 'Multiplayer lobby' });
  const style = e('style', { text: lobbyStyles });
  const status = e('span', { class: 'lobby-connection', 'data-testid': 'lobby-connection', role: 'status' });
  const error = e('div', { class: 'lobby-error', role: 'alert', 'data-testid': 'lobby-error', hidden: true });
  const button = (text, action, attrs = {}) => e('button', { type: 'button', 'data-action': action, ...attrs, text });
  const solo = button('Play solo', 'solo', { class: 'lobby-quiet', 'data-testid': 'lobby-solo' });
  const reconnect = button('Reconnect', 'reconnect', { class: 'lobby-quiet', 'data-testid': 'lobby-reconnect', hidden: true });
  const hero = e('header', { class: 'lobby-hero' },
    e('div', {}, e('div', { class: 'lobby-eyebrow', text: 'Human Descent · Multiplayer' }),
      e('h1', { text: 'Find your descent.' }),
      e('p', { class: 'lobby-intro', text: 'Desktop co-op: meet in chat, bring your team, and steer a shared model toward a lower loss.' })),
    e('div', { class: 'lobby-hero-actions' }, status, reconnect, solo));
  const nameId = `${mountId}-name`;
  const nameInput = e('input', { id: nameId, name: 'name', maxlength: 48, autocomplete: 'nickname', required: true, placeholder: 'How should we call you?', 'data-testid': 'lobby-name' });
  nameInput.value = client.savedName;
  const enterButton = e('button', { type: 'submit', class: 'lobby-primary', 'data-testid': 'lobby-enter', text: 'Enter lobby' });
  const newSession = button('Start a new session', 'new-session', { class: 'lobby-primary', 'data-testid': 'lobby-new-session', hidden: true });
  const entry = e('section', { class: 'lobby-entry', 'data-testid': 'lobby-entry' },
    e('h2', { text: 'A shared model. A different perspective.' }),
    e('p', { text: 'Play together on desktop with one to three people per team. Each player controls one landscape; every move helps the same model.' }),
    e('form', { 'data-form': 'identity' }, e('label', { for: nameId, text: 'Display name' }), nameInput,
      e('div', { class: 'lobby-entry-actions' }, enterButton, newSession)));
  const left = e('aside', { class: 'lobby-column lobby-left', 'data-testid': 'lobby-team-area' });
  const chatLog = e('div', { class: 'lobby-chat-log', role: 'log', 'aria-label': 'Lobby messages', 'aria-live': 'polite', 'aria-relevant': 'additions', 'data-testid': 'lobby-chat-log' });
  const chatInputId = `${mountId}-message`;
  const chatInput = e('textarea', { id: chatInputId, name: 'message', rows: 2, maxlength: 500, required: true, placeholder: 'Say hello to the room…', 'data-testid': 'lobby-chat-input' });
  const send = e('button', { type: 'submit', class: 'lobby-primary', text: 'Send', 'data-testid': 'lobby-chat-send' });
  const chatPanel = e('section', { class: 'lobby-panel lobby-chat' },
    e('div', { class: 'lobby-panel-heading' }, e('h2', { text: 'The common room' }), e('span', { class: 'lobby-badge', text: 'Shared chat' })),
    chatLog,
    e('form', { class: 'lobby-chat-compose', 'data-form': 'chat' }, e('label', { for: chatInputId, text: 'Message the room' }), chatInput,
      e('div', { class: 'lobby-chat-footer' }, e('span', { class: 'lobby-muted', text: 'Enter to send · Shift + Enter for a new line' }), send)));
  const center = e('div', { class: 'lobby-column lobby-center' }, chatPanel);
  const right = e('aside', { class: 'lobby-column lobby-right' });
  const grid = e('div', { class: 'lobby-grid', hidden: true }, left, center, right);
  host.append(style, hero, error, entry, grid);
  root.replaceChildren(host);

  function panel(title, testid, children, extra = '') {
    return e('section', { class: `lobby-panel ${extra}`, 'data-testid': testid },
      e('div', { class: 'lobby-panel-heading' }, e('h2', { text: title })),
      e('div', { class: 'lobby-panel-content' }, children));
  }
  function badge(text, ready = false) { return e('span', { class: `lobby-badge${ready ? ' is-ready' : ''}`, text }); }
  function note(text) { return e('p', { class: 'lobby-description', text }); }
  function action(text, command, attrs = {}) {
    return button(text, command, { disabled: client.state.connection !== 'connected' || client.state.pendingCount > 0, ...attrs });
  }
  function person(player, sub, extra) {
    return e('div', { class: 'lobby-person' },
      e('span', { class: 'lobby-avatar', 'aria-hidden': true, text: (player?.name || '?').slice(0, 1).toLocaleUpperCase() }),
      e('div', { class: 'lobby-person-detail' }, e('span', { class: 'lobby-person-name', text: `${player?.name || 'Player'}${player?.id === currentView.you?.id ? ' (you)' : ''}` }),
        e('span', { class: 'lobby-person-sub', text: sub })), extra);
  }
  function memberList(team) {
    return e('ul', { class: 'lobby-roster' }, team.memberIds.map(id => {
      const player = currentView.players.get(id);
      const sub = [id === team.captainId ? 'Captain' : 'Teammate', player?.connected ? 'online' : 'reconnecting'].join(' · ');
      return e('li', {}, person(player, sub));
    }));
  }
  function teamName(id) { return currentView.teams.get(id)?.name || 'Team'; }
  function matchLabel(match) {
    return match.teamIds.map(id => {
      const team = currentView.teams.get(id);
      const size = team?.memberIds.length || 1;
      return `${teamName(id)} · ${size} ${size === 1 ? 'player' : 'players'}`;
    }).join(' vs ');
  }
  function draftInput(testid, attrs = {}) {
    const input = e('input', { ...attrs, 'data-testid': testid, 'data-draft': testid });
    if (drafts.has(testid)) input.value = drafts.get(testid);
    return input;
  }

  function renderOwnTeam(view) {
    if (!view.team) {
      const inputId = `${mountId}-team`;
      return panel('Your team', 'lobby-your-team', [note('Create a team, then invite up to two players from the room. Team captains choose the name and opponents.'),
        e('form', { 'data-form': 'create-team' }, e('label', { for: inputId, text: 'Team name' }),
          draftInput('lobby-team-name', { id: inputId, name: 'teamName', required: true, maxlength: 48, placeholder: 'Local Minima' }),
          e('div', { class: 'lobby-button-row' }, e('button', { type: 'submit', class: 'lobby-primary', disabled: !canAct(), text: 'Create team', 'data-testid': 'lobby-create-team' }))),
        note('Already have an invitation? Accept it below to join its team.')]);
    }
    const team = view.team;
    const idle = team.status === 'available';
    const children = [e('div', { class: 'lobby-team-top' }, e('h3', { text: team.name, 'data-testid': 'lobby-own-team-name' }), badge(idle ? `${team.memberIds.length}/3 players` : 'In a match', !idle)), memberList(team)];
    if (team.memberIds.length < 3) children.push(e('div', { class: 'lobby-slot', text: view.captain ? 'Invite a player from the room to fill a landscape.' : 'Your captain can invite more players.' }));
    if (view.captain) {
      const inputId = `${mountId}-rename`;
      children.push(e('form', { 'data-form': 'rename-team', class: 'lobby-inline-form' },
        e('div', { class: 'lobby-field' }, e('label', { for: inputId, text: 'Team name' }),
          draftInput('lobby-rename-name', { id: inputId, name: 'teamName', required: true, maxlength: 48, value: team.name, disabled: !idle })),
        e('button', { type: 'submit', text: 'Save', disabled: !canAct() || !idle, 'data-testid': 'lobby-rename-team' })));
      children.push(e('div', { class: 'lobby-stack', 'data-testid': 'lobby-match-settings' },
        e('h3', { text: 'Next match settings' }),
        e('label', { for: `${mountId}-batch-size`, text: 'Batch size' }),
        draftInput('lobby-batch-size', { id: `${mountId}-batch-size`, type: 'number', min: 1, max: 512, step: 1, value: 32, disabled: !idle }),
        e('label', { for: `${mountId}-max-dimensions`, text: 'Maximum dimensions (blank = unlimited)' }),
        draftInput('lobby-max-dimensions', { id: `${mountId}-max-dimensions`, type: 'number', min: 6, max: 6006, step: 6, placeholder: 'Unlimited', disabled: !idle }),
        note('Includes the starting 6 directions. Each switch adds 6: a limit of 18 allows 2 switches. These settings apply to both teams and lock when the match is created.')));
      children.push(e('div', { class: 'lobby-button-row' }, action('Start team speed run', 'team.start_speedrun', { class: 'lobby-primary', disabled: !canAct() || !idle, 'data-testid': 'lobby-start-speedrun' })));
    } else children.push(note('Your captain starts speed runs and chooses challenges.'));
    children.push(e('div', { class: 'lobby-button-row' }, action(view.captain ? 'Disband team' : 'Leave team', 'team.leave', { class: 'lobby-quiet lobby-danger', disabled: !canAct() || !idle, 'data-testid': 'lobby-leave-team' })));
    return panel('Your team', 'lobby-your-team', e('div', { class: 'lobby-stack' }, children));
  }

  function renderInvitations(view) {
    const received = view.invitations.filter(i => i.recipientId === view.you.id);
    const sent = view.invitations.filter(i => i.inviterId === view.you.id);
    if (!received.length && !sent.length) return null;
    return panel('Invitations', 'lobby-invitations', [
      ...received.map(invite => e('div', { class: 'lobby-invitation', 'data-testid': `lobby-invitation-${invite.id}` },
        e('p', {}, e('strong', { text: teamName(invite.teamId) }), ' invited you to play.'),
        e('div', { class: 'lobby-button-row' },
          action('Accept', 'invite.accept', { class: 'lobby-primary', 'data-id': invite.id, 'data-testid': `lobby-invite-accept-${invite.id}` }),
          action('Decline', 'invite.decline', { 'data-id': invite.id, 'data-testid': `lobby-invite-decline-${invite.id}` })))),
      ...sent.map(invite => e('div', { class: 'lobby-invitation' }, e('p', { text: `Invitation sent to ${view.players.get(invite.recipientId)?.name || 'player'}. Waiting for their reply.` }))),
    ]);
  }

  function renderChallenges(view) {
    if (!view.team) return null;
    const challenges = view.challenges.filter(c => c.targetTeamId === view.team.id || c.challengerTeamId === view.team.id);
    if (!challenges.length) return null;
    return panel('Team challenges', 'lobby-challenges', challenges.map(challenge => {
      const incoming = challenge.targetTeamId === view.team.id;
      const opponent = incoming ? challenge.challengerTeamId : challenge.targetTeamId;
      const opponentSize = view.teams.get(opponent)?.memberIds.length || 1;
      const actions = view.captain ? e('div', { class: 'lobby-button-row' },
        ...(incoming ? [
          action('Accept challenge', 'challenge.accept', { class: 'lobby-primary', 'data-id': challenge.id, 'data-testid': `lobby-challenge-accept-${challenge.id}` }),
          action('Decline', 'challenge.decline', { 'data-id': challenge.id, 'data-testid': `lobby-challenge-decline-${challenge.id}` }),
        ] : [action('Cancel challenge', 'challenge.cancel', { 'data-id': challenge.id, 'data-testid': `lobby-challenge-cancel-${challenge.id}` })])) : note('Your captain will respond to this challenge.');
      return e('div', { class: 'lobby-invitation' }, e('h3', { text: teamName(opponent) }),
        e('p', { class: 'lobby-muted', text: `${view.team.memberIds.length} vs ${opponentSize} · ${incoming ? 'Incoming challenge' : 'Challenge sent'}` }),
        e('p', { text: settingsLabel(challenge.settings), 'data-testid': 'lobby-challenge-settings' }), actions);
    }));
  }

  function renderMatch(view) {
    const match = view.activeMatch;
    if (!match) return null;
    const roster = match.teamIds.flatMap(id => view.teams.get(id)?.memberIds || []);
    const ready = match.readyPlayerIds.includes(view.you.id);
    const preparing = match.phase === 'preparing';
    const checking = ['ready', 'preparing'].includes(match.phase);
    const phaseLabel = preparing ? 'Preparing your landscapes' : match.phase === 'ready' ? 'Ready for the descent?' : match.phase === 'countdown' ? 'The countdown is on' : 'Run in progress';
    const children = [e('h3', { text: phaseLabel }), e('p', { class: 'lobby-muted', text: matchLabel(match) }),
      e('p', { text: settingsLabel(match.settings), 'data-testid': 'lobby-confirmed-settings' }),
      note(preparing ? 'The server is setting up a shared model for each team.' : 'Each player controls one landscape. The lowest final validation loss wins.'),
      e('div', { class: 'lobby-match-count', 'data-testid': 'lobby-ready-count', text: `${match.readyPlayerIds.length}/${roster.length} ready` }),
      e('ul', { class: 'lobby-roster' }, roster.map(id => e('li', {}, person(view.players.get(id), teamName(view.players.get(id)?.teamId), badge(match.readyPlayerIds.includes(id) ? 'Ready' : 'Waiting', match.readyPlayerIds.includes(id))))))];
    if (checking) children.push(e('div', { class: 'lobby-button-row' },
      action(ready ? 'Not ready' : 'I’m ready', 'match.ready', { class: ready ? '' : 'lobby-primary', disabled: !canAct() || preparing, 'data-id': match.id, 'data-ready': String(!ready), 'data-testid': 'lobby-ready' }),
      view.captain ? action('Cancel run', 'match.cancel', { class: 'lobby-quiet', 'data-id': match.id, 'data-testid': 'lobby-cancel-match' }) : null));
    children.push(note(checking ? 'All participants must be ready before the game opens. Teams stay together after the run.' : 'Your team stays together here when the run finishes.'));
    return panel('Next run', 'lobby-match', children, 'lobby-match');
  }

  function renderResults(view) {
    const match = view.lastResult;
    if (!match) return null;
    const finished = match.phase === 'finished';
    const results = [...(match.results || [])].sort((a, b) => (a.loss ?? Infinity) - (b.loss ?? Infinity));
    const valid = results.length === match.teamIds.length && results.every(r => Number.isFinite(r.loss));
    const tied = results.length > 1 && valid && results.every(r => r.loss === results[0].loss);
    const resultTitle = !finished ? (match.phase === 'failed' ? 'Run failed' : 'Run cancelled') : match.teamIds.length === 1 ? 'Team run complete' : tied ? 'A tie. Well played.' : 'Match complete';
    const children = [e('h3', { text: resultTitle }), e('p', { class: 'lobby-muted', text: matchLabel(match) }), note(settingsLabel(match.settings))];
    if (finished && valid) {
      children.push(...results.map((result, index) => e('div', { class: 'lobby-result-row', 'data-testid': `lobby-result-${result.teamId}` },
        e('span', { text: `${results.length > 1 && !tied && index === 0 ? 'Winner · ' : ''}${teamName(result.teamId)}` }),
        e('strong', { text: result.loss.toFixed(6) }))));
      children.push(note('Final validation loss · lower is better. Team results stay separate from solo scores.'));
    } else children.push(note(match.reason || (finished ? 'No valid final score is available.' : 'Team reservations have been released. You can start another run.')));
    children.push(e('p', { class: 'lobby-muted', text: 'Your team is still together. Chat, invite, or start another descent.' }));
    return panel('Latest result', 'lobby-results', children);
  }

  function renderTeams(view) {
    const cards = [...view.teams.values()].map(team => {
      const own = team.id === view.team?.id;
      const canChallenge = view.captain && !own && view.team.status === 'available' && team.status === 'available';
      const children = [e('div', { class: 'lobby-team-top' }, e('h3', { text: team.name }), badge(own ? 'Your team' : team.status === 'busy' ? 'In a match' : `${team.memberIds.length}/3`)), memberList(team)];
      if (canChallenge) children.push(action(`Challenge · ${view.team.memberIds.length} vs ${team.memberIds.length}`, 'challenge.create', { 'data-id': team.id, 'data-testid': `lobby-challenge-${team.id}` }));
      else if (!own) children.push(e('p', { class: 'lobby-muted', text: 'Membership is by captain invitation.' }));
      return e('article', { class: `lobby-team-card${own ? ' is-own' : ''}`, 'data-testid': `lobby-team-${team.id}` }, children);
    });
    return panel(`Teams · ${view.teams.size}`, 'lobby-teams', cards.length ? e('div', { class: 'lobby-teams' }, cards) : note('The first team starts with you. Create one and invite a player from the room.'));
  }

  function renderPresence(view) {
    const players = [...view.players.values()].sort((a, b) => Number(b.connected) - Number(a.connected) || a.name.localeCompare(b.name));
    const inviteAllowed = view.captain && view.team.status === 'available' && view.team.memberIds.length < 3;
    const online = players.filter(p => p.connected).length;
    return panel(`In the room · ${online} online`, 'lobby-presence', e('div', { class: 'lobby-presence-list' }, players.map(player => {
      const invited = view.invitations.some(i => i.recipientId === player.id && i.teamId === view.team?.id);
      const invite = inviteAllowed && player.id !== view.you.id && !player.teamId
        ? action(invited ? 'Invited' : 'Invite', 'team.invite', { 'data-id': player.id, disabled: !canAct() || invited, 'data-testid': `lobby-invite-${player.id}` }) : null;
      return person(player, `${player.teamId ? teamName(player.teamId) : 'No team'}${player.connected ? '' : ' · reconnecting'}`, invite);
    })));
  }

  let lastChatKey = '';
  function renderChat(snapshot) {
    const key = JSON.stringify(snapshot.chat);
    if (key === lastChatKey) return;
    const atBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 70;
    const oldScroll = chatLog.scrollTop;
    const initial = lastChatKey === '';
    lastChatKey = key;
    if (!snapshot.chat.length) {
      chatLog.replaceChildren(e('div', { class: 'lobby-chat-empty' }, e('strong', { text: 'A good run starts with a hello.' }), e('span', { text: 'This room is shared by every team.' })));
      return;
    }
    const messages = snapshot.chat.map(message => {
      const sent = message.sentAt == null ? null : new Date(message.sentAt * 1000);
      const time = sent && !Number.isNaN(sent.valueOf()) ? sent.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
      return e('article', { class: `lobby-message${message.playerId === snapshot.you.id ? ' is-you' : ''}`, 'data-testid': 'lobby-chat-message' },
        e('div', { class: 'lobby-message-head' }, e('span', { class: 'lobby-message-name', text: message.name }), e('time', { class: 'lobby-message-time', datetime: sent?.toISOString(), text: time })),
        e('p', { class: 'lobby-message-text', text: message.text }));
    });
    chatLog.replaceChildren(...messages);
    chatLog.scrollTop = initial || atBottom ? chatLog.scrollHeight : oldScroll;
  }

  function canAct() { return client.state.connection === 'connected' && client.state.pendingCount === 0; }
  function saveDrafts() {
    for (const input of host.querySelectorAll('[data-draft]')) drafts.set(input.getAttribute('data-draft'), input.value);
  }
  function render(state) {
    if (destroyed) return;
    saveDrafts();
    const focused = host.contains(doc.activeElement) ? doc.activeElement : null;
    const focusId = focused?.getAttribute('data-testid');
    const selection = focused && typeof focused.selectionStart === 'number' ? [focused.selectionStart, focused.selectionEnd] : null;
    const labels = { idle: 'Choose a name', connecting: 'Connecting…', identifying: 'Signing in…', connected: 'Connected', reconnecting: 'Reconnecting…', 'identity-rejected': 'Session needs attention', error: 'Connection needs attention', closed: 'Disconnected' };
    status.textContent = labels[state.connection] || state.connection;
    status.dataset.status = state.connection;
    reconnect.hidden = !['error', 'identity-rejected'].includes(state.connection);
    reconnect.textContent = state.connection === 'identity-rejected' ? 'Try session again' : 'Reconnect';
    error.hidden = !state.error;
    error.textContent = state.error?.message || '';
    const rejected = state.connection === 'identity-rejected';
    entry.hidden = !!state.snapshot && !rejected;
    enterButton.hidden = rejected;
    newSession.hidden = !rejected;
    const connecting = ['connecting', 'identifying', 'reconnecting'].includes(state.connection);
    enterButton.disabled = connecting;
    enterButton.textContent = connecting ? 'Connecting…' : 'Enter lobby';
    nameInput.disabled = connecting;
    grid.hidden = !state.snapshot || rejected;
    chatInput.disabled = state.connection !== 'connected';
    send.disabled = !canAct();
    if (!state.snapshot) return;
    currentView = deriveLobby(state.snapshot);
    left.replaceChildren(...[renderMatch(currentView), renderOwnTeam(currentView), renderInvitations(currentView), renderChallenges(currentView), renderResults(currentView)].filter(Boolean));
    right.replaceChildren(renderPresence(currentView), renderTeams(currentView), e('div', { class: 'lobby-muted lobby-notes' },
      e('p', { text: 'Teams have 1–3 players. Captains invite; recipients accept or decline.' }),
      e('p', { text: 'Unequal team sizes are welcome. Each team shares one model, with one landscape per player.' }),
      e('p', { text: 'Captains choose batch size and dimension limits before a match. The server enforces them, the countdown, and final scores.' })));
    renderChat(state.snapshot);
    if (focused && !host.contains(focused) && focusId) {
      const replacement = [...host.querySelectorAll('[data-testid]')].find(n => n.getAttribute('data-testid') === focusId);
      if (replacement && !replacement.disabled) {
        replacement.focus({ preventScroll: true });
        if (selection && typeof replacement.setSelectionRange === 'function') replacement.setSelectionRange(...selection);
      }
    }
  }

  async function run(type, payload = {}, success) {
    try {
      const result = await client.command(type, payload);
      if (!destroyed) success?.(result);
    } catch { /* The correlated server error is already in client.state. */ }
  }
  function onSubmit(event) {
    const form = event.target.closest('form[data-form]');
    if (!form || !host.contains(form)) return;
    event.preventDefault();
    if (form.dataset.form === 'identity') { client.connect(nameInput.value); return; }
    if (!canAct()) return;
    const type = form.dataset.form;
    if (type === 'chat') {
      const text = chatInput.value.trim();
      if (text) run('chat.send', { text }, () => { if (chatInput.value.trim() === text) chatInput.value = ''; });
    } else if (type === 'create-team' || type === 'rename-team') {
      const input = form.querySelector('input');
      const name = input.value.trim();
      if (name) run(type === 'create-team' ? 'team.create' : 'team.rename', { name }, () => {
        drafts.delete(input.getAttribute('data-draft'));
        if (type === 'rename-team') render(client.state);
      });
    }
  }
  function onClick(event) {
    const target = event.target.closest('[data-action]');
    if (!target || !host.contains(target) || target.disabled) return;
    const type = target.dataset.action;
    if (type === 'solo') { onSolo(); return; }
    if (type === 'reconnect') { client.connect(); return; }
    if (type === 'new-session') { client.resetIdentity(); client.connect(nameInput.value); return; }
    if (!canAct()) return;
    const id = target.dataset.id;
    if (type === 'team.invite') run(type, { playerId: id });
    else if (type.startsWith('invite.')) run(type, { inviteId: id });
    else if (['challenge.accept', 'challenge.decline', 'challenge.cancel'].includes(type)) run(type, { challengeId: id });
    else if (type === 'challenge.create' || type === 'team.start_speedrun') {
      try {
        const batch = host.querySelector('[data-testid="lobby-batch-size"]').value;
        const dimensions = host.querySelector('[data-testid="lobby-max-dimensions"]').value.trim();
        const settings = normalizeSettings({ batchSize: Number(batch), maxDimensions: dimensions === '' ? null : Number(dimensions) });
        run(type, { ...(type === 'challenge.create' ? { teamId: id } : {}), settings });
      } catch (err) { error.textContent = err.message; error.hidden = false; }
    }
    else if (type === 'match.ready') run(type, { matchId: id, ready: target.dataset.ready === 'true' });
    else if (type === 'match.cancel') run(type, { matchId: id });
    else if (type === 'team.leave') run(type);
  }
  function onKey(event) {
    // This host is the complete keyboard boundary. Text editing and native
    // buttons keep their defaults; gameplay's document shortcuts cannot fire.
    event.stopPropagation();
    if (event.type === 'keydown' && event.target === chatInput && event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!event.repeat && canAct()) chatInput.form.requestSubmit();
    }
  }
  host.addEventListener('submit', onSubmit);
  host.addEventListener('click', onClick);
  host.addEventListener('keydown', onKey);
  host.addEventListener('keyup', onKey);
  host.addEventListener('keypress', onKey);
  const unsubscribeLaunch = client.subscribeLaunch(descriptor => { if (!destroyed) onLaunch(descriptor); });
  const unsubscribe = client.subscribe(render);
  if (client.savedName) client.connect();

  return {
    client,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      unsubscribe();
      unsubscribeLaunch();
      host.removeEventListener('submit', onSubmit);
      host.removeEventListener('click', onClick);
      host.removeEventListener('keydown', onKey);
      host.removeEventListener('keyup', onKey);
      host.removeEventListener('keypress', onKey);
      client.destroy();
      host.remove();
    },
  };
}
