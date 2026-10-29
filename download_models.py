"""Download the two pretrained files DriveSafe needs (~55 MB total).

YuNet  - face detection (OpenCV model zoo)
LBF    - 68 facial landmarks (OpenCV contrib facemark)

Note on certificates: the python.org macOS installer ships without root
certificates, so a plain urllib call fails with CERTIFICATE_VERIFY_FAILED.
We use certifi's bundle when it is installed and fall back to curl, which
uses the system keychain.
"""
import shutil
import ssl
import subprocess
import sys
import urllib.request

from config import LBF_PATH, MODELS_DIR, YUNET_PATH

FILES = [
    (YUNET_PATH,
     "https://github.com/opencv/opencv_zoo/raw/main/models/"
     "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    (LBF_PATH,
     "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"),
]

MIN_BYTES = 100_000


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _via_urllib(url, path) -> bool:
    try:
        ctx = _ssl_context()
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(path, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:5.1f} / {total / 1e6:5.1f} MB", end="")
        print()
        return True
    except Exception as err:
        print(f"\r  urllib failed ({type(err).__name__}), trying curl")
        return False


def _via_curl(url, path) -> bool:
    if not shutil.which("curl"):
        return False
    try:
        subprocess.run(["curl", "-L", "--fail", "--progress-bar", "-o", str(path), url],
                       check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in FILES:
        if path.exists() and path.stat().st_size > MIN_BYTES:
            print(f"[ok]   {path.name} already here")
            continue

        print(f"[get]  {path.name}")
        ok = _via_urllib(url, path) or _via_curl(url, path)

        if not ok or not path.exists() or path.stat().st_size < MIN_BYTES:
            path.unlink(missing_ok=True)
            sys.exit(
                f"\nCould not download {path.name}.\n"
                f"Download it by hand and put it in models/:\n  {url}\n"
            )
        print(f"[done] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
