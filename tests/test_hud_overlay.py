import os

import pygame as pg

from hudes.hud_overlay import CONTROL_GROUPS, render_bottom_hud


def test_render_bottom_hud_surface(tmp_path):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pg.init()
    pg.font.init()

    status_text = "val:1.23 bs:64 (f32) t:12.3s dims:12 sgd:99"
    surface = render_bottom_hud(status_text, CONTROL_GROUPS)

    assert surface.get_width() > 200
    assert surface.get_height() > 60

    # ensure there is alpha content (not fully transparent)
    alpha_array = pg.surfarray.array_alpha(surface)
    assert (alpha_array > 0).any()

    pg.quit()
