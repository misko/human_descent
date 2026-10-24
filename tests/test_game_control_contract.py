"""The lobby boundary remains usable without model/inference dependencies."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from hudes.game_control import MatchPreset, MatchSpec, PlayerSlot, TeamRoster


def test_match_contract_passes_immutable_roster_values():
    roster = TeamRoster("blue", "Blue", (PlayerSlot("alice", "Alice", 0),))
    spec = MatchSpec("match-1", (roster,))
    with pytest.raises(FrozenInstanceError):
        spec.teams[0].players[0].panel_index = 2
    with pytest.raises(FrozenInstanceError):
        spec.preset.duration_seconds = 1
    assert spec.preset.scoring == "final_validation_loss_v2"
    assert MatchPreset().duration_seconds == 120


def test_game_control_import_has_no_inference_dependencies():
    code = """
import importlib.abc
import sys
class BlockInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'numpy', 'pygame'}:
            raise AssertionError('Inference dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockInference())
from hudes.game_control import MatchSpec, TeamRoster, PlayerSlot
assert MatchSpec('match', (TeamRoster('team', 'Team', (PlayerSlot('p', 'P', 0),)),)).match_id == 'match'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
