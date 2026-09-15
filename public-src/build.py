"""从 YAML 同步公开类别并打包，同时更新资源版本，避免浏览器使用旧脚本。"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]  # 与当前工作目录无关。


# 类别参数只维护在 YAML；示例目录匹配大小写后保留实际目录名。
def main():
    config = yaml.safe_load((ROOT / "ee_video_viewer.yaml").read_text())
    config = config["profiles"]
    if not config or "H01" not in config:
        raise ValueError("ee_video_viewer.yaml 必须包含 profiles.H01 作为公开版默认类别")
    profiles = {}
    for name, item in config.items():
        matches = [p for p in (ROOT / "public/example").iterdir() if p.is_dir() and p.name.casefold() == name.casefold()]
        if len(matches) > 1:
            raise ValueError(f"类别 {name} 对应多个示例目录")
        profiles[name] = {
            "label": name, "exampleRoot": f"example/{matches[0].name if matches else name}",
            "videoKey": item["video_key"], "armsKey": item["arms_key"],
            "indices": item["lr_xyz_indices"], "direction": item["lr_xyz_direction"],
        }
    # 临时入口文件不进入仓库，输出 bundle 自带类别配置，不依赖 Python 服务。
    source = (ROOT / "public-src/viewer.js").read_text()
    source = source.replace("const PROFILES = PUBLIC_PROFILES", "const PROFILES = " + json.dumps(profiles, ensure_ascii=False))
    source = source.replace("const DEFAULT_PROFILE = PUBLIC_DEFAULT_PROFILE", "const DEFAULT_PROFILE = " + json.dumps(next(iter(profiles))))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", dir=ROOT / "public-src", encoding="utf-8") as entry:
        entry.write(source)
        entry.flush()
        subprocess.run([str(ROOT / "public-src/node_modules/.bin/esbuild"), entry.name,
                        "--bundle", "--minify", "--format=iife", "--outfile=" + str(ROOT / "public/assets/h01-viewer-public.js")], check=True)
    page = ROOT / "public/index.html"
    html = page.read_text()
    for name in ("h01-viewer-public.js", "viewer-core.js"):
        version = hashlib.sha256((ROOT / "public/assets" / name).read_bytes()).hexdigest()[:12]
        html = re.sub(r'assets/' + re.escape(name) + r'(?:\?v=[^" ]*)?', f"assets/{name}?v={version}", html)
    page.write_text(html)
    print("公开类别已同步：" + ", ".join(profiles))


if __name__ == "__main__":
    main()
