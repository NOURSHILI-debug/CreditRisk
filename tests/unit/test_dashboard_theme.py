from __future__ import annotations

from dashboard.components import theme


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return sum(value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722)))


def _contrast_ratio(foreground: str, background: str) -> float:
    first, second = _luminance(foreground), _luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def test_normal_text_tokens_meet_wcag_aa_contrast_on_page_background() -> None:
    for foreground in (theme.INK, theme.MUTED, theme.BLUE, theme.TEAL, theme.AMBER, theme.RED):
        assert _contrast_ratio(foreground, theme.BACKGROUND) >= 4.5


def test_sidebar_text_token_meets_wcag_aa_contrast_on_navy() -> None:
    assert _contrast_ratio("#F2F5F8", theme.NAVY) >= 4.5
