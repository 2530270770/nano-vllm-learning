"""Application-wide font selection and visual theme."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from PySide6.QtGui import QFont, QFontDatabase


UI_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Noto Sans SC",
    "SimHei",
    "DengXian",
    "Segoe UI",
)

COLORS = {
    "background": "#F3F6FB",
    "surface": "#FFFFFF",
    "surface_muted": "#F8FAFD",
    "border": "#DDE5F0",
    "text": "#182033",
    "muted": "#68758A",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_soft": "#EAF1FF",
    "success": "#16805B",
    "success_soft": "#E8F7F0",
    "warning": "#B56A11",
    "warning_soft": "#FFF4E5",
    "danger": "#C43D4B",
    "danger_soft": "#FDECEF",
}

PROJECT_FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
LOCAL_FONT_DIR = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
)
SYSTEM_FONT_FILES = (
    Path(os.environ.get("NANO_VLLM_CJK_FONT", "")),
    PROJECT_FONT_DIR / "NotoSansSC-Regular.ttf",
    PROJECT_FONT_DIR / "NotoSansCJKsc-Regular.otf",
    PROJECT_FONT_DIR / "SourceHanSansSC-Regular.otf",
    PROJECT_FONT_DIR / "msyh.ttc",
    WINDOWS_FONT_DIR / "msyh.ttc",
    WINDOWS_FONT_DIR / "msyhbd.ttc",
    WINDOWS_FONT_DIR / "Deng.ttf",
    WINDOWS_FONT_DIR / "simhei.ttf",
    LOCAL_FONT_DIR / "msyh.ttc",
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    Path("/mnt/d/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
)

_fonts_registered = False
_plot_font_family = "DejaVu Sans"
_plot_has_cjk_font = False
CJK_FAMILY_MARKERS = (
    "microsoft yahei",
    "noto sans cjk",
    "noto sans sc",
    "source han sans",
    "wenquanyi",
    "pingfang",
    "simhei",
)


def _register_system_fonts():
    """Register a CJK font with both Qt and Matplotlib."""
    global _fonts_registered, _plot_font_family, _plot_has_cjk_font
    if _fonts_registered:
        return

    for font in font_manager.fontManager.ttflist:
        name = font.name.casefold()
        if not any(marker in name for marker in CJK_FAMILY_MARKERS):
            continue
        if not font.fname or not Path(font.fname).is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(font.fname)
        if font_id < 0:
            continue
        _plot_font_family = font.name
        _plot_has_cjk_font = True
        break

    for font_path in () if _plot_has_cjk_font else SYSTEM_FONT_FILES:
        if not str(font_path) or not font_path.is_file():
            continue
        try:
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id < 0:
                continue
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
        except Exception:
            continue
        if family:
            _plot_font_family = family
            _plot_has_cjk_font = True
            break

    _fonts_registered = True


def available_font_families() -> list[str]:
    return list(QFontDatabase.families())


def choose_ui_font_family() -> str:
    """Pick the first installed font with reliable CJK coverage."""
    _register_system_fonts()
    families = available_font_families()
    by_lower = {family.casefold(): family for family in families}
    for candidate in UI_FONT_CANDIDATES:
        if candidate.casefold() in by_lower:
            return by_lower[candidate.casefold()]
    return QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()


def plot_font_supports_cjk() -> bool:
    """Return whether Matplotlib has a real CJK font available."""
    _register_system_fonts()
    return _plot_has_cjk_font


def apply_theme(app) -> str:
    """Apply a consistent font to Qt and Matplotlib, then style the app."""
    family = choose_ui_font_family()
    fallback_families = [family]
    for candidate in UI_FONT_CANDIDATES:
        if candidate.casefold() not in {item.casefold() for item in fallback_families}:
            fallback_families.append(candidate)

    font = QFont()
    font.setFamilies(fallback_families)
    font.setFamily(family)
    font.setPointSize(10)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)

    _register_system_fonts()
    plot_families = [_plot_font_family]
    if "DejaVu Sans".casefold() not in {item.casefold() for item in plot_families}:
        plot_families.append("DejaVu Sans")
    mpl.rcParams.update({
        "font.family": plot_families,
        "font.sans-serif": plot_families,
        "axes.unicode_minus": False,
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"],
    })
    app.setStyleSheet(STYLESHEET)
    return family


STYLESHEET = """
QWidget {
    color: #182033;
}
QMainWindow, QWidget#AppRoot {
    background: #F3F6FB;
}
QFrame#HeaderCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #14233D, stop:0.58 #193A67, stop:1 #2458A2);
    border: 1px solid #2F63AA;
    border-radius: 18px;
}
QLabel#AppTitle {
    color: #FFFFFF;
    font-size: 21pt;
    font-weight: 700;
}
QLabel#AppSubtitle {
    color: #C9D9F0;
    font-size: 9.5pt;
}
QLabel#StatusPill {
    color: #DCE8F8;
    background: rgba(255, 255, 255, 28);
    border: 1px solid rgba(255, 255, 255, 45);
    border-radius: 13px;
    padding: 5px 12px;
    font-weight: 600;
}
QLabel#StatusPill[state="running"] {
    color: #FFF1D6;
    background: rgba(255, 184, 76, 34);
    border-color: rgba(255, 201, 118, 85);
}
QLabel#StatusPill[state="success"] {
    color: #DDF8EC;
    background: rgba(48, 196, 139, 34);
    border-color: rgba(110, 226, 181, 85);
}
QLabel#StatusPill[state="error"] {
    color: #FFE5E8;
    background: rgba(230, 77, 94, 34);
    border-color: rgba(255, 144, 154, 85);
}
QFrame#Card, QGroupBox#Card {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 14px;
}
QGroupBox#Card {
    margin-top: 13px;
    padding: 14px 14px 12px 14px;
    font-weight: 700;
}
QGroupBox#Card::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 5px;
    color: #26364F;
    background: #FFFFFF;
}
QLabel#SectionTitle {
    color: #253550;
    font-size: 11pt;
    font-weight: 700;
}
QLabel#SectionHint, QLabel#MutedLabel {
    color: #748198;
}
QLabel#MetricTitle {
    color: #6A778D;
    font-size: 8.5pt;
    font-weight: 600;
}
QLabel#MetricValue {
    color: #182033;
    font-size: 17pt;
    font-weight: 700;
}
QLabel#MetricDetail {
    color: #8792A6;
    font-size: 8pt;
}
QFrame#MetricCard {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 12px;
}
QLineEdit, QSpinBox {
    min-height: 34px;
    padding: 0 10px;
    background: #F8FAFD;
    border: 1px solid #D7E0EB;
    border-radius: 8px;
    selection-background-color: #BFD5FF;
}
QLineEdit:hover, QSpinBox:hover {
    border-color: #B8C7DA;
    background: #FFFFFF;
}
QLineEdit:focus, QSpinBox:focus {
    background: #FFFFFF;
    border: 1px solid #3B82F6;
}
QSpinBox {
    padding-right: 22px;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 20px;
    border: 0;
    background: transparent;
}
QSpinBox::up-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #728096;
}
QSpinBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #728096;
}
QPushButton {
    min-height: 35px;
    padding: 0 14px;
    color: #31435D;
    background: #F8FAFD;
    border: 1px solid #D4DEEA;
    border-radius: 8px;
    font-weight: 600;
}
QPushButton:hover {
    color: #1F3E6E;
    background: #EEF4FC;
    border-color: #B7C8DD;
}
QPushButton:pressed {
    background: #E3ECF8;
}
QPushButton:disabled {
    color: #A2ABBA;
    background: #F1F4F8;
    border-color: #E4E9F0;
}
QPushButton[role="primary"] {
    color: #FFFFFF;
    background: #2563EB;
    border-color: #2563EB;
}
QPushButton[role="primary"]:hover {
    color: #FFFFFF;
    background: #1D4ED8;
    border-color: #1D4ED8;
}
QPushButton[role="primary"]:pressed {
    background: #1E40AF;
}
QPushButton[role="danger"] {
    color: #B93646;
    background: #FFF7F8;
    border-color: #F0C9CF;
}
QPushButton[role="danger"]:hover {
    color: #9F2535;
    background: #FDECEF;
    border-color: #E9AAB4;
}
QFrame#ResultSummary {
    background: #F6F9FE;
    border: 1px solid #DCE6F3;
    border-radius: 9px;
}
QLabel#ResultText {
    color: #4D5D73;
    padding: 9px;
}
QSplitter::handle {
    background: transparent;
    width: 10px;
}
QTabWidget::pane {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 12px;
    top: -1px;
}
QTabBar::tab {
    min-width: 100px;
    min-height: 35px;
    padding: 0 16px;
    margin-right: 4px;
    color: #718096;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabBar::tab:selected {
    color: #245DCC;
    border-bottom-color: #2563EB;
}
QTabBar::tab:hover:!selected {
    color: #41536C;
}
QPlainTextEdit {
    color: #D9E5F4;
    background: #111A2B;
    border: 0;
    border-radius: 10px;
    padding: 12px;
    selection-background-color: #2B5BA5;
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
}
QScrollArea {
    background: transparent;
    border: 0;
}
QScrollBar:vertical {
    width: 10px;
    margin: 2px;
    background: transparent;
}
QScrollBar::handle:vertical {
    min-height: 28px;
    background: #C6D1E0;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #A9B8CC;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QToolTip {
    color: #FFFFFF;
    background: #24344D;
    border: 1px solid #3C5272;
    padding: 5px;
}
"""
