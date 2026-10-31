"""公式フォントを実行時に取得し、ハッシュとライセンスを確認する。"""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "295d98a7a0c17c68f1341eaeea354e7960ea70d3"
BASE = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl/notosansjp/"
FILES = {
    "NotoSansJP-VF.ttf": ("NotoSansJP%5Bwght%5D.ttf", "c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f"),
    "OFL.txt": ("OFL.txt", "1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9"),
}

def main():
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    for name, (remote, expected) in FILES.items():
        path = assets / name
        data = path.read_bytes() if path.is_file() else urllib.request.urlopen(BASE + remote, timeout=90).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("公式配布ファイルのハッシュが一致しません。処理を中止します。")
        path.write_bytes(data)
    print("フォントとライセンスのハッシュを確認しました。")

if __name__ == "__main__":
    main()
