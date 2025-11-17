export const HELP_TOUR_SCREENS = ['tour_hook', 'tour_modes', 'tour_keys'];

export const HELP_TAGLINE = 'You are the optimizer.';

export const HELP_ELEVATOR_PITCH =
    'Steer a CNN through its loss landscape—walk along random directions in weight space, find valleys and saddles, and set a 2-minute MNIST speed-run record.';

export const HOW_IT_WORKS_POINTS = [
    'The cyan grid shows one random 2D slice of the CNN weights. Each point is a full neural network.',
    'Height corresponds to training loss. Walking downhill means lowering the loss.',
    'Batch controls, precision, and eval toggles let you inspect how the network behaves in different regimes.',
    'Speed-Run Mode locks in a 2-minute countdown and submits your best validation score to the global leaderboard.',
];

export const TOUR_MODE_OPTIONS = [
    {
        id: 'explore',
        title: 'Explore Mode',
        tagline: 'Learn & tinker',
        description: 'Sample random directions, inspect batches, and develop intuition.',
    },
    {
        id: 'speed',
        title: 'Speed-Run Mode',
        tagline: '2-minute challenge',
        description: 'Ranked leaderboard. Configure quickly and keep loss low.',
        nudge: 'Try Ranked later',
    },
];

export const MOBILE_KEY_COMMANDS = [
    { keys: 'Single-finger drag', label: 'Move in plane' },
    { keys: 'Two-finger drag', label: 'Rotate camera' },
    { keys: 'Buttons', label: 'Batch size / Step size / Precision' },
    { keys: 'Tap grid', label: 'Select cyan mesh' },
    { keys: 'Speed 🔥', label: 'Speed-Run toggle' },
];
