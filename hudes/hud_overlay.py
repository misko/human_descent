from typing import Iterable, Sequence

import pygame as pg

CONTROL_GROUPS: Sequence[dict] = (
    {"icon": None, "keys": ("W", "A", "S", "D"), "label": "Move"},
    {"icon": None, "keys": ("⬆️", "⬇️", "⬅️", "➡️"), "label": "Rotate"},
    {"icon": None, "keys": ("Spacebar",), "label": "New Dims"},
    {"icon": None, "keys": ("Enter",), "label": "New Batch"},
    {"icon": None, "keys": ("⇧",), "label": "Cycle Plane"},
    {"icon": None, "keys": ("[", "]"), "label": "Step ±"},
    {"icon": None, "keys": (";",), "label": "Batch-size"},
    {"icon": None, "keys": ("'",), "label": "FP16/32"},
    {"icon": None, "keys": ("X",), "label": "Help"},
    {"icon": None, "keys": ("Q",), "label": "Q: Hold to quit"},
)


def _ensure_font_init() -> None:
    if not pg.get_init():
        pg.init()
    if not pg.font.get_init():
        pg.font.init()


def _render_text(
    text: str, font: pg.font.Font, color: tuple[int, int, int, int]
) -> pg.Surface:
    if text is None:
        text = ""
    return font.render(text, True, color)


def _render_pill(text: str, font: pg.font.Font) -> pg.Surface:
    label = _render_text(text, font, (255, 255, 255, 255))
    pad_x, pad_y = 4, 3
    width = label.get_width() + pad_x * 2
    height = label.get_height() + pad_y * 2
    surface = pg.Surface((width, height), pg.SRCALPHA)

    rect = surface.get_rect()
    pg.draw.rect(surface, (255, 255, 255, 30), rect, border_radius=height // 2)
    pg.draw.rect(
        surface,
        (255, 255, 255, 75),
        rect.inflate(-1, -1),
        width=1,
        border_radius=height // 2,
    )

    label_rect = label.get_rect(center=rect.center)
    surface.blit(label, label_rect)
    return surface


def _horizontal_stack(surfaces: Iterable[pg.Surface], gap: int) -> pg.Surface:
    surfaces = tuple(surfaces)
    if not surfaces:
        return pg.Surface((0, 0), pg.SRCALPHA)
    width = sum(surface.get_width() for surface in surfaces) + gap * (len(surfaces) - 1)
    height = max(surface.get_height() for surface in surfaces)
    stacked = pg.Surface((width, height), pg.SRCALPHA)
    x = 0
    for surface in surfaces:
        stacked.blit(surface, (x, (height - surface.get_height()) // 2))
        x += surface.get_width() + gap
    return stacked


def _render_group(
    icon: str | None,
    keys: Sequence[str],
    label_text: str,
    key_font: pg.font.Font,
    label_font: pg.font.Font,
) -> pg.Surface:
    elements: list[pg.Surface] = []
    if icon:
        elements.append(_render_text(icon, label_font, (255, 255, 255, 255)))

    key_surfaces = tuple(_render_pill(key, key_font) for key in keys)
    if key_surfaces:
        elements.append(_horizontal_stack(key_surfaces, gap=2))

    if label_text:
        elements.append(_render_text(label_text, label_font, (200, 240, 255, 225)))

    return _horizontal_stack(elements, gap=6)


def render_bottom_hud(
    status_text: str,
    control_groups: Sequence[dict] = CONTROL_GROUPS,
) -> pg.Surface:
    """Create a pygame surface representing the bottom HUD overlay."""
    _ensure_font_init()

    status_font = pg.font.SysFont("Consolas", 20)
    key_font = pg.font.SysFont("Consolas", 18, bold=True)
    label_font = pg.font.SysFont("Consolas", 18)

    status_surface = _render_text(
        status_text or "",
        status_font,
        (200, 230, 255, 235),
    )

    group_surfaces = [
        _render_group(
            group.get("icon"),
            group.get("keys", ()),
            group.get("label", ""),
            key_font,
            label_font,
        )
        for group in control_groups
    ]

    separators = [
        _render_text("•", label_font, (190, 220, 255, 170))
        for _ in range(len(group_surfaces) - 1)
    ]

    control_elements: list[pg.Surface] = []
    for group_surface, sep_surface in zip(
        group_surfaces,
        separators + [None],
    ):
        control_elements.append(group_surface)
        if sep_surface is not None:
            control_elements.append(sep_surface)

    controls_surface = _horizontal_stack(control_elements, gap=8)

    blocks = [
        block for block in (status_surface, controls_surface) if block.get_width() > 0
    ]

    max_width = max(block.get_width() for block in blocks)
    padding_x = 16
    padding_y = 10
    inner_gap = 6 if len(blocks) > 1 else 0
    total_height = (
        padding_y * 2
        + sum(block.get_height() for block in blocks)
        + inner_gap * (len(blocks) - 1)
    )

    hud_surface = pg.Surface(
        (max_width + padding_x * 2, total_height),
        pg.SRCALPHA,
    )
    rect = hud_surface.get_rect()

    # Draw translucent background
    bg_color = (12, 18, 30, 190)
    border_color = (86, 198, 255, 96)
    pg.draw.rect(hud_surface, bg_color, rect, border_radius=18)
    pg.draw.rect(
        hud_surface,
        border_color,
        rect.inflate(-2, -2),
        width=2,
        border_radius=16,
    )

    y = padding_y
    for index, block in enumerate(blocks):
        hud_surface.blit(
            block,
            ((hud_surface.get_width() - block.get_width()) // 2, y),
        )
        y += block.get_height()
        if index < len(blocks) - 1:
            y += inner_gap

    return hud_surface
