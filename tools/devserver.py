"""开发用静态服务器：http.server + 正确的 .mjs/.wasm MIME 类型 + 无头验证上报。

用法：python tools/devserver.py [端口] [目录]
（Python 内置 http.server 不认识 .mjs，浏览器会因 MIME 拒绝 ES module——
本脚本补上映射后即可托管 mobile/ 端侧页面；另提供 POST /report 接收
无头浏览器验证页的完成上报，写入 headless_report.json。）
"""

import functools
import http.server
import mimetypes
import sys

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".js")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/report":
            n = int(self.headers.get("content-length", 0))
            with open("headless_report.json", "wb") as f:
                f.write(self.rfile.read(n))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # 静音请求日志
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 7866
_DIR = sys.argv[2] if len(sys.argv) > 2 else "."

if __name__ == "__main__":
    http.server.test(Handler, port=port, bind="127.0.0.1")
