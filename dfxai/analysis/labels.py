"""
그림 라벨 언어 설정 (영문 / 한글).

    from .labels import T, set_language
    set_language("ko")          # 이후 T("English", "한글") 은 "한글" 을 돌려줌

한글을 고르면 matplotlib 기본 글꼴을 Malgun Gothic(없으면 Noto Sans KR)으로
바꾸고 음수 기호 깨짐(unicode_minus)을 막습니다. 국내 대회 원고는 한글 라벨,
영문 학술지는 영문 라벨을 쓰면 되므로 두 벌을 모두 만들 수 있게 했습니다.
"""
from __future__ import annotations
import matplotlib
from matplotlib import font_manager

_LANG = "en"
_KO_FONTS = ("Malgun Gothic", "Noto Sans KR", "NanumGothic", "Gulim", "Dotum")


def set_language(lang: str) -> None:
    global _LANG
    _LANG = "ko" if str(lang).lower().startswith("ko") else "en"
    if _LANG == "ko":
        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in _KO_FONTS:
            if name in avail:
                matplotlib.rcParams["font.family"] = name
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        matplotlib.rcParams["axes.unicode_minus"] = True


def language() -> str:
    return _LANG


def T(en: str, ko: str) -> str:
    """현재 언어의 라벨."""
    return ko if _LANG == "ko" else en
