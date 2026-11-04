"""
한글(HWP 2018 이상) 자동화로 hwpx 를 열어 쪽수를 확인하고 PDF(와 선택적으로 HWP)로 저장한다. Windows 전용.

    python -m dfxai.analysis.hwp_export 원고.hwpx [--pdf 출력.pdf] [--hwp 출력.hwp]

자동화로 한글 경로의 새 파일을 열면 보안 확인 창이 숨은 채 떠서 멈추는 일이 있었다. 그래서 파일을 임시 폴더의
영문 이름으로 복사해 열고, 결과도 임시 폴더에 저장한 뒤 지정한 경로로 옮긴다.
"""
from __future__ import annotations
import argparse, os, shutil, sys, tempfile


def export(src: str, pdf: str | None = None, hwp: str | None = None) -> int:
    import win32com.client as win32
    import win32com.client.dynamic  # noqa: F401
    tmp = tempfile.mkdtemp(prefix="hwpexp_")
    s = os.path.join(tmp, "in.hwpx"); shutil.copy(src, s)
    app = win32.dynamic.Dispatch("HWPFrame.HwpObject")   # gen_py 캐시가 깨져도 동작하도록 동적 바인딩
    try:
        app.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    except Exception:
        pass
    try:
        app.XHwpWindows.Item(0).Visible = False
    except Exception:
        pass
    app.SetMessageBoxMode(0x00020021)
    if not app.Open(s, "HWPX", "forceopen:true"):
        raise RuntimeError("한글이 파일을 열지 못함")
    pages = app.PageCount
    if pdf:
        app.SaveAs(os.path.join(tmp, "out.pdf"), "PDF", ""); shutil.copy(os.path.join(tmp, "out.pdf"), pdf)
    if hwp:
        app.SaveAs(os.path.join(tmp, "out.hwp"), "HWP", ""); shutil.copy(os.path.join(tmp, "out.hwp"), hwp)
    app.Clear(1); app.Quit()
    shutil.rmtree(tmp, ignore_errors=True)
    return pages


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--pdf"); ap.add_argument("--hwp")
    a = ap.parse_args()
    print("pages:", export(a.src, a.pdf, a.hwp), flush=True)


if __name__ == "__main__":
    main()
