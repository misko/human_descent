export const lobbyStyles = `
.hd-lobby { --lobby-ink:#e9edf8; --lobby-muted:#9caac1; --lobby-line:rgba(161,182,220,.18); --lobby-accent:#b2f5d3; color:var(--lobby-ink); font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; width:min(1440px,100%); margin:0 auto; padding:30px 28px 42px; box-sizing:border-box; text-align:left; }
.hd-lobby *, .hd-lobby *::before, .hd-lobby *::after { box-sizing:border-box; }
.hd-lobby [hidden] { display:none!important; }
.hd-lobby h1,.hd-lobby h2,.hd-lobby h3,.hd-lobby p { margin:0; }
.hd-lobby button,.hd-lobby input,.hd-lobby textarea { font:inherit; }
.hd-lobby button { min-height:38px; padding:8px 13px; border:1px solid var(--lobby-line); border-radius:9px; background:#23314b; color:var(--lobby-ink); cursor:pointer; transition:background .15s,border-color .15s; box-shadow:none; position:static; width:auto; height:auto; text-transform:none; letter-spacing:0; }
.hd-lobby button:hover:not(:disabled) { background:#314360; border-color:#7390b2; }
.hd-lobby button:disabled { opacity:.42; cursor:default; }
.hd-lobby button.lobby-primary { color:#0d2c25; background:var(--lobby-accent); border-color:var(--lobby-accent); font-weight:700; }
.hd-lobby button.lobby-primary:hover:not(:disabled) { background:#d0ffe7; border-color:#d0ffe7; }
.hd-lobby button.lobby-quiet { background:transparent; }
.hd-lobby button.lobby-danger { color:#ffc9c6; }
.hd-lobby :focus-visible { outline:2px solid var(--lobby-accent); outline-offset:3px; }
.hd-lobby input,.hd-lobby textarea { display:block; width:100%; min-width:0; padding:10px 12px; border:1px solid #40506b; border-radius:9px; color:var(--lobby-ink); background:#101a2c; outline-offset:2px; resize:vertical; }
.hd-lobby input::placeholder,.hd-lobby textarea::placeholder { color:#8291aa; }
.hd-lobby label { display:block; color:var(--lobby-muted); font-size:12px; font-weight:650; margin:0 0 7px; }
.hd-lobby .lobby-hero { display:flex; justify-content:space-between; align-items:flex-start; gap:22px; padding-bottom:26px; }
.hd-lobby .lobby-eyebrow { color:var(--lobby-accent); font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.16em; margin-bottom:7px; }
.hd-lobby h1 { font-size:clamp(26px,4vw,38px); font-weight:720; line-height:1.2; letter-spacing:-.04em; }
.hd-lobby .lobby-intro { max-width:620px; color:var(--lobby-muted); margin-top:9px; }
.hd-lobby .lobby-hero-actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
.hd-lobby .lobby-connection { display:flex; align-items:center; gap:7px; color:var(--lobby-muted); white-space:nowrap; font-size:12px; }
.hd-lobby .lobby-connection::before { content:''; width:7px; height:7px; border-radius:50%; background:#d9b76c; }
.hd-lobby .lobby-connection[data-status="connected"]::before { background:var(--lobby-accent); box-shadow:0 0 10px #b2f5d330; }
.hd-lobby .lobby-error { padding:12px 15px; border:1px solid #b9616070; background:#442627; color:#ffd9d6; border-radius:10px; margin:0 0 18px; overflow-wrap:anywhere; }
.hd-lobby .lobby-entry { max-width:550px; padding:26px; margin:28px auto; border:1px solid var(--lobby-line); border-radius:16px; background:#162136; }
.hd-lobby .lobby-entry h2 { font-size:23px; margin-bottom:9px; }
.hd-lobby .lobby-entry p { color:var(--lobby-muted); margin-bottom:22px; }
.hd-lobby .lobby-entry-actions { display:flex; gap:9px; margin-top:14px; flex-wrap:wrap; }
.hd-lobby .lobby-grid { display:grid; grid-template-columns:minmax(250px,300px) minmax(300px,1fr) minmax(225px,270px); gap:18px; align-items:start; }
.hd-lobby .lobby-column { display:flex; flex-direction:column; gap:18px; min-width:0; }
.hd-lobby .lobby-panel { min-width:0; border:1px solid var(--lobby-line); background:linear-gradient(145deg,#1c293e,#152035); border-radius:14px; overflow:hidden; }
.hd-lobby .lobby-panel-heading { padding:17px 18px; border-bottom:1px solid var(--lobby-line); display:flex; justify-content:space-between; align-items:center; gap:10px; }
.hd-lobby h2 { font-size:14px; font-weight:750; }
.hd-lobby h3 { font-size:15px; font-weight:700; overflow-wrap:anywhere; }
.hd-lobby .lobby-panel-content { padding:17px 18px; }
.hd-lobby .lobby-muted { color:var(--lobby-muted); font-size:12px; }
.hd-lobby .lobby-description { color:var(--lobby-muted); margin:9px 0 15px; font-size:13px; }
.hd-lobby .lobby-inline-form { display:flex; gap:8px; align-items:end; }
.hd-lobby .lobby-inline-form .lobby-field { flex:1; min-width:0; }
.hd-lobby .lobby-stack { display:flex; flex-direction:column; gap:10px; }
.hd-lobby .lobby-button-row { display:flex; gap:7px; flex-wrap:wrap; margin-top:12px; }
.hd-lobby .lobby-badge { display:inline-block; border:1px solid var(--lobby-line); border-radius:6px; padding:2px 7px; color:var(--lobby-muted); font-size:10px; font-weight:700; white-space:nowrap; }
.hd-lobby .lobby-badge.is-ready { color:var(--lobby-accent); border-color:#b2f5d355; background:#b2f5d30c; }
.hd-lobby .lobby-roster { list-style:none; margin:14px 0; padding:0; display:flex; flex-direction:column; gap:10px; }
.hd-lobby .lobby-person { display:flex; align-items:center; gap:9px; min-width:0; }
.hd-lobby .lobby-avatar { display:grid; place-items:center; border:1px solid #7e9ad740; background:#5675a625; width:30px; height:30px; border-radius:9px; flex:none; color:#c5d5f3; font-size:12px; font-weight:750; }
.hd-lobby .lobby-person-detail { flex:1; min-width:0; overflow-wrap:anywhere; }
.hd-lobby .lobby-person-name { display:block; font-size:13px; font-weight:600; }
.hd-lobby .lobby-person-sub { display:block; color:var(--lobby-muted); font-size:10px; }
.hd-lobby .lobby-slot { padding:9px 11px; border:1px dashed #7990b035; border-radius:9px; color:#8292ad; font-size:11px; }
.hd-lobby .lobby-teams { display:grid; gap:12px; }
.hd-lobby .lobby-team-card { padding:14px; border:1px solid var(--lobby-line); border-radius:11px; background:#132035; }
.hd-lobby .lobby-team-card.is-own { border-color:#b2f5d355; }
.hd-lobby .lobby-team-top { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; }
.hd-lobby .lobby-team-card .lobby-roster { margin:12px 0; }
.hd-lobby .lobby-invitation { padding:13px; border:1px solid #91a7cb40; border-radius:10px; background:#17273d; }
.hd-lobby .lobby-invitation + .lobby-invitation { margin-top:10px; }
.hd-lobby .lobby-invitation p { font-size:12px; overflow-wrap:anywhere; }
.hd-lobby .lobby-chat { display:flex; flex-direction:column; min-height:440px; }
.hd-lobby .lobby-chat-log { height:clamp(260px,43vh,600px); overflow-y:auto; overscroll-behavior:contain; padding:18px; display:flex; flex-direction:column; gap:18px; scrollbar-color:#465b78 transparent; }
.hd-lobby .lobby-chat-empty { display:grid; place-content:center; flex:1; text-align:center; color:var(--lobby-muted); }
.hd-lobby .lobby-chat-empty strong { color:#bdcce2; font-weight:600; display:block; margin-bottom:6px; }
.hd-lobby .lobby-message { overflow-wrap:anywhere; }
.hd-lobby .lobby-message-head { display:flex; align-items:baseline; gap:9px; }
.hd-lobby .lobby-message-name { font-weight:700; color:#dbe7ff; font-size:12px; }
.hd-lobby .lobby-message.is-you .lobby-message-name { color:var(--lobby-accent); }
.hd-lobby .lobby-message-time { color:#7f90ac; font-size:10px; white-space:nowrap; }
.hd-lobby .lobby-message-text { margin-top:3px; font-size:13px; color:#c6d2e5; white-space:pre-wrap; }
.hd-lobby .lobby-chat-compose { padding:14px 18px; border-top:1px solid var(--lobby-line); }
.hd-lobby .lobby-chat-compose textarea { min-height:67px; max-height:170px; resize:vertical; }
.hd-lobby .lobby-chat-footer { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-top:9px; }
.hd-lobby .lobby-presence-list { max-height:440px; overflow-y:auto; display:flex; flex-direction:column; gap:15px; }
.hd-lobby .lobby-presence-list .lobby-person button { min-height:29px; padding:4px 8px; font-size:11px; }
.hd-lobby .lobby-match { border-color:#b2f5d355; }
.hd-lobby .lobby-match h3 { margin-bottom:6px; }
.hd-lobby .lobby-match .lobby-match-count { color:var(--lobby-accent); font-weight:750; margin-top:12px; }
.hd-lobby .lobby-result-row { display:flex; justify-content:space-between; gap:10px; padding:10px 0; border-bottom:1px solid var(--lobby-line); font-size:12px; }
.hd-lobby .lobby-result-row span:first-child { overflow-wrap:anywhere; }
.hd-lobby .lobby-result-row strong { font-variant-numeric:tabular-nums; }
.hd-lobby .lobby-notes { padding:0 2px; }
.hd-lobby .lobby-notes p + p { margin-top:9px; }
@media(max-width:1120px) { .hd-lobby .lobby-grid { grid-template-columns:minmax(235px,290px) minmax(300px,1fr); } .hd-lobby .lobby-right { grid-column:1/-1; display:grid; grid-template-columns:1fr 1fr; } .hd-lobby .lobby-presence-list { max-height:270px; } }
@media(max-width:670px) { .hd-lobby { padding:20px 14px 32px; } .hd-lobby .lobby-hero { flex-direction:column; gap:16px; } .hd-lobby .lobby-hero-actions { justify-content:flex-start; } .hd-lobby .lobby-grid { display:flex; flex-direction:column; } .hd-lobby .lobby-column { width:100%; } .hd-lobby .lobby-center { order:1; } .hd-lobby .lobby-left { order:0; } .hd-lobby .lobby-right { display:flex; order:2; } .hd-lobby .lobby-chat-log { height:300px; } .hd-lobby .lobby-entry { margin:10px auto; padding:22px; } }
`;
