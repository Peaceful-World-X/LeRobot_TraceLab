#!/usr/bin/env python3
"""启动支持视频分段读取的公开查看器：python serve_public.py 19100 --directory public。"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """按需传输文件，浏览器取消旧视频属于正常连接结束。"""

    # 客户端切换类别、跳帧或关闭标签时，不打印正常断连的堆栈。
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    # 同样处理响应收尾期间的客户端断连，其他异常仍正常上报。
    def finish(self):
        try:
            super().finish()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # 解析单段 Range，支持前缀、开放结束和后缀请求。
    @staticmethod
    def parse_range(header, size):
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip()) if len(header) < 200 else None
        if not match or not any(match.groups()) or size <= 0:
            raise ValueError("invalid byte range")
        first, last = match.groups()
        if not first:
            count = int(last)
            if count <= 0:
                raise ValueError("invalid suffix range")
            return max(0, size - count), size - 1
        start, end = int(first), int(last) if last else size - 1
        if start >= size or end < start:
            raise ValueError("unsatisfiable byte range")
        return start, min(end, size - 1)

    # 对文件提供 HEAD、ETag 和 206；目录沿用标准索引页面行为。
    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(Path(self.directory).resolve()):
            self.send_error(403, "Path outside served directory")
            return None
        if path.is_dir():
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        try:
            stat = os.fstat(stream.fileno())
            size = stat.st_size
            etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
            modified = self.date_time_string(stat.st_mtime)
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                stream.close()
                return None
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range")
            if_range = self.headers.get("If-Range")
            if self.command == "GET" and requested and (not if_range or if_range in (etag, modified)):
                try:
                    start, end = self.parse_range(requested, size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    stream.close()
                    return None
                status = 206
            self.remaining = max(0, end - start + 1)
            stream.seek(start)
            self.send_response(status)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(self.remaining))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", modified)
            self.send_header("Cache-Control", "no-cache")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            return stream
        except Exception:
            stream.close()
            raise

    # 每次最多读取 64 KiB，且只发送客户端请求的字节区间。
    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        while self.remaining:
            block = source.read(min(64 * 1024, self.remaining))
            if not block:
                break
            outputfile.write(block)
            self.remaining -= len(block)


# 保留与 python -m http.server 相近的调用方式，默认服务项目 public 目录。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", type=int, default=19100)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "public")
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")
    try:
        server = ThreadingHTTPServer((args.bind, args.port), partial(RangeRequestHandler, directory=str(root)))
    except OSError as exc:
        parser.exit(1, f"无法监听 {args.bind}:{args.port}：{exc}；请先停止旧服务或更换端口。\n")
    with server:
        print(f"Range 服务已启动：http://{args.bind}:{args.port}，目录：{root}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
