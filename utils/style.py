"""
matplotlib 한글 폰트 자동 감지 및 스타일 설정
Colab / 로컬 환경 모두 지원
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform, subprocess, os


def _find_korean_font():
    """시스템에서 한글 폰트를 자동으로 찾아 반환"""
    # 1) Colab: fonts-nanum 설치 시도
    try:
        if os.path.exists("/content"):  # Colab 환경
            subprocess.run(
                ["apt-get", "-qq", "install", "-y", "fonts-nanum"],
                capture_output=True, timeout=30,
            )
            fm._load_fontmanager(try_read_cache=False)  # 폰트 캐시 갱신
    except Exception:
        pass

    # 2) 등록된 폰트에서 한글 폰트 검색 (우선순위 순)
    candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "Apple SD Gothic Neo", "Noto Sans KR",
        "Paperlogy", "D2Coding",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name

    # 3) ttf 파일 직접 탐색
    search_dirs = [
        os.path.expanduser("~/tmp_fonts"),
        os.path.expanduser("~/fonts"),
        "/usr/share/fonts",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".ttf") and ("nanum" in f.lower() or "gothic" in f.lower()):
                    path = os.path.join(root, f)
                    fm.fontManager.addfont(path)
                    prop = fm.FontProperties(fname=path)
                    return prop.get_name()

    return None  # 한글 폰트 없음


def set_style(korean=True):
    """matplotlib 스타일을 논문 품질로 설정"""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    if korean:
        font_name = _find_korean_font()
        if font_name:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✓ 한글 폰트 설정: {font_name}")
        else:
            print("⚠ 한글 폰트를 찾지 못했습니다. 기본 폰트를 사용합니다.")


# 논문 색상 팔레트
COLORS = {
    "navy": "#2C3E50",
    "teal": "#1B7A8A",
    "amber": "#F39C12",
    "green": "#27AE60",
    "coral": "#E74C3C",
}
