export const BRAND_TAGLINE = 'You are the optimizer.';
export const SPLASH_LOGO_URL = '/hudes_logo_splash.jpg';
export const SPLASH_MEDIA_URL = '/loss_landscape.gif';

export const WELCOME_BULLETS = [
    'Steer a CNN through weight space—walk along random directions, find valleys and saddles, and set a 2-minute MNIST speed-run record.',
    'Each point is a neural network; moving downhill lowers the loss.',
    'Desktop gives you full control and the fastest runs.',
];

export const HOW_IT_WORKS_POINTS = [
    '<strong>Parameterization:</strong> Flatten all weights to a vector <code>w ∈ ℝ<sup>P</sup></code>. Current network is <code>w</code>; loss is <code>L(w)</code>.',
    '<strong>Random directions:</strong> Sample orthogonal directions in <code>w ∈ ℝ<sup>P</sup></code>. “New dims” generates fresh random orthogonal directions.',
    '<strong>1D move:</strong> <code>w&apos; = w + α·d₁</code>, where <code>α = step_size × t</code>, <code>t ∈ ℤ</code>. The Z-plot charts <code>L(w&apos;)</code>.',
    '<strong>2D move:</strong> <code>w&apos; = w + α·d₁ + β·d₂</code>. WASD controls <code>(α, β)</code>; the mesh shows <code>(α, β) ↦ L(w&apos;)</code>.',
    '<strong>New directions:</strong> Space samples fresh <code>d₁, d₂</code> at the current <code>w</code>; Shift + click cycles the cyan plane. Mobile runs one plane at a time.',
    '<strong>Loss:</strong> Enter (or the Eval button on mobile) runs a full validation pass for leaderboard scoring and updates the confusion matrix.',
    '<strong>Step size / zoom:</strong> <code>[ ]</code> scales <code>α, β</code>; larger steps explore faster but can overshoot minima.',
    '<strong>Precision:</strong> <code>&apos;</code> toggles FP16/FP32 to trade speed for numerical stability.',
];

export const MODE_CARDS = [
    {
        id: 'explore',
        title: 'Explore Mode',
        eyebrow: 'Learn & tinker',
        description: 'Sample random directions, adjust step size, batch size, and precision.',
        primaryCta: 'Continue',
        defaultSelected: true,
    },
    {
        id: 'speed',
        title: 'Speed-Run Mode',
        eyebrow: '2-minute ranked challenge',
        description: 'Lowest validation loss joins the global leaderboard. Desktop recommended.',
        primaryCta: 'Start Ranked',
    },
];

export const DESKTOP_CONTROLS_GRID = [
    { label: 'Move', value: 'W / A / S / D / Scroll' },
    { label: 'Rotate', value: 'Arrow keys / click' },
    { label: 'New random directions', value: 'Space' },
    { label: 'Toggle plane', value: 'Shift / Click grid' },
    { label: 'Step ± (zoom)', value: '[ ]' },
    { label: 'Batch-size', value: ';' },
    { label: 'Precision FP16/32', value: '\'' },
    { label: 'Full eval (validation)', value: 'Enter' },
    { label: 'Speed-Run', value: 'Z' },
    { label: 'Help', value: '?' },
];

export const MOBILE_CONTROLS_GRID = [
    { label: 'Move', value: 'Single-finger drag' },
    { label: 'Rotate', value: 'Two-finger drag' },
    { label: 'New random directions', value: 'Tap Space button' },
    { label: 'Step ± / Batch / Precision', value: 'Bottom buttons' },
    { label: 'Full eval (validation)', value: 'Eval button' },
    { label: 'Speed-Run', value: 'Speed 🔥 button' },
    { label: 'Help', value: 'Tap ?' },
];

export const TUTORIAL_TASKS = [
    {
        id: 'move',
        title: 'Move',
        detailDesktop: 'W/A/S/D or scroll along the cyan plane.',
        detailMobile: 'Drag with one finger to move in-plane.',
    },
    {
        id: 'rotate',
        title: 'Rotate view',
        detailDesktop: 'Arrow keys or click to tilt the camera.',
        detailMobile: 'Use two fingers to rotate the view.',
    },
    {
        id: 'new_dims',
        title: 'New directions',
        detailDesktop: 'Press Space for fresh random directions.',
        detailMobile: 'Tap the New Dims button.',
    },
    {
        id: 'step',
        title: 'Adjust step size',
        detailDesktop: 'Use [ and ] to zoom in or out.',
        detailMobile: 'Use the Step +/- buttons.',
    },
    {
        id: 'batch',
        title: 'Change batch size',
        detailDesktop: 'Tap ; to cycle the batch size.',
        detailMobile: 'Tap Batch Cycle to change batch size.',
    },
    {
        id: 'precision',
        title: 'Toggle precision',
        detailDesktop: 'Press \' to switch FP16 / FP32.',
        detailMobile: 'Tap the FP button to toggle precision.',
    },
];

export const SPEEDRUN_RULES = [
    'Fixed starting seed per run (fairness).',
    'Any batch size and precision.',
    'Space to sample new subspaces; Shift + click to cycle subspace.',
    'Enter for new train batch and full eval.',
    'Auto eval at end of Speed-Run.',
];

export const SPEEDRUN_COMPACT_KEYS =
    'Move WASD • Rotate ←↑→↓ / drag • New directions Space • Cycle plane Shift + click • Step ± [ ] • Batch ; • Precision \' • Eval Enter • Help ?';

export const SPEEDRUN_RULES_MOBILE = [
    'Fixed starting seed per run (fairness).',
    'Any batch size or precision (use the bottom buttons).',
    'Tap DIMS for new subspaces.',
    'Tap BATCH for new train batch and full eval.',
    'Auto eval at end of Speed-Run.',
];

export const SPEEDRUN_COMPACT_KEYS_MOBILE =
    'Move drag • Rotate two-finger • New dims Space button • Step/Batch/FP bottom buttons • Eval button • Help ?';

export const TOUR_FLOWS = {
    initial: ['tour_splash', 'tour_welcome', 'tour_modes'],
    explore: ['tour_canvas'],
    speed: ['tour_speed'],
};

export const TOUR_SCREENS = Array.from(
    new Set(Object.values(TOUR_FLOWS).flat()),
);

export const SHARE_TEXT = (valLoss, evalSteps, planes) => {
    const loss = typeof valLoss === 'number' && Number.isFinite(valLoss) ? valLoss.toFixed(4) : '—';
    const evalPart = evalSteps ? `${evalSteps} eval-steps` : 'eval-steps pending';
    const planePart = planes ? `${planes} planes` : 'planes pending';
    return `Descended MNIST to ${loss} in 2:00—can you beat it? ${evalPart} ${planePart}`;
};
