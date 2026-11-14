"""手机摄像头连拍服务：手机浏览器打开局域网页面抓帧 → 上传合并。

与 app.py（PC 端）相互独立，可同时运行：
  .venv\\Scripts\\python.exe phone.py            # 默认端口 7861
  .venv\\Scripts\\python.exe phone.py --port 7861 --no-https

说明：
- 浏览器 getUserMedia（摄像头）要求安全上下文：局域网 IP 访问必须 HTTPS，
  因此默认生成并使用自签证书（首次运行自动生成到 .certs/，手机浏览器提示
  "不安全"时点「高级 -> 继续访问」即可）；
- 终端会打印局域网地址 + 二维码，手机同 Wi-Fi 扫码即开；
- 抓帧与成片归档到 data/real_burst/<时间戳>_phone/，与 PC 端 Tab 3 一致。
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gpmerge import merge_burst  # noqa: E402
from gpmerge.landmarks import DEFAULT_MODEL_PATH, Landmarker  # noqa: E402
from gpmerge.pipeline import save_report  # noqa: E402

CERT_DIR = ROOT / ".certs"
CERT_FILE = CERT_DIR / "phone.pem"
KEY_FILE = CERT_DIR / "phone.key"

_LM: Landmarker | None = None


def _landmarker() -> Landmarker:
    global _LM
    if _LM is None:
        _LM = Landmarker(DEFAULT_MODEL_PATH)
    return _LM


def _b64_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


app = FastAPI(title="gpmerge phone burst")


@app.get("/")
def index() -> str:
    return (ROOT / "phone_page.html").read_text("utf-8")


@app.post("/api/merge")
async def api_merge(request: Request) -> dict:
    # 注意：不用 FastAPI 的声明式 File(...) —— 它与 starlette 1.10 组合存在
    # multipart 解析失效问题（实测恒 422），starlette 原生 request.form() 正常。
    form = await request.form()
    uploads = sorted([v for v in form.getlist("frames") if hasattr(v, "filename")],
                     key=lambda x: x.filename or "")
    if not uploads:
        raise HTTPException(400, "没有收到连拍帧")

    burst_dir = ROOT / "data" / "real_burst" / f"{time.strftime('%Y%m%d_%H%M%S')}_phone"
    burst_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploads:
        data = await f.read()
        if not data:
            continue
        p = burst_dir / Path(f.filename or f"frame_{len(paths):03d}.jpg").name
        p.write_bytes(data)
        paths.append(p)
    if len(paths) < 3:
        raise HTTPException(400, "有效连拍帧不足 3 张")

    try:
        res = merge_burst(paths, landmarker=_landmarker())
    except AssertionError as e:
        raise HTTPException(400, f"读图失败：{e}")
    except RuntimeError as e:
        raise HTTPException(422, f"{e}（请确认合影对象都在镜头内、光线充足）")
    save_report(res, paths, burst_dir)

    rows = [["P" + str(r.person_id),
             "保留" if r.action == "kept" else
             f"替换 ← 第{r.src_frame}帧" if r.action == "swapped" else
             f"补回 ← 第{r.src_frame}帧",
             f"{r.score_base:.3f}", f"{r.score_src:.3f}",
             f"{r.ear_base:.3f}", f"{r.ear_src:.3f}"] for r in res.rows]
    info = (f"基准底片：{paths[res.base_frame_idx].name}（{len(paths)} 帧输入），"
            f"替换 {sum(1 for r in res.rows if r.action != 'kept')} 人 / "
            f"共 {len(res.rows)} 人。已归档 data/real_burst/{burst_dir.name}")
    return {"merged": _b64_jpeg(res.merged),
            "annotated": _b64_jpeg(res.annotated),
            "info": info, "rows": rows}


def lan_ips() -> list[str]:
    """枚举本机所有候选 IPv4（排除回环/链路本地），普通局域网网段排前。

    机器可能有多网卡（VPN/Tailscale 虚拟网卡会干扰单地址探测），全部列出
    让用户挑能打开的那个；证书 SAN 覆盖全部候选。
    """
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))     # 不发包，仅取默认路由源地址
        addrs.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    def rank(ip: str) -> int:
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if any(ip.startswith(f"172.{i}.") for i in range(16, 32)):
            return 2
        if ip.startswith("100."):      # CGNAT/VPN 网段（Tailscale 等）
            return 5
        return 3

    return sorted((a for a in addrs
                   if not a.startswith("127.") and not a.startswith("169.254.")),
                  key=rank)


def ensure_cert() -> tuple[Path, Path]:
    """生成自签证书（10 年，SAN 含 localhost + 全部候选 IP）。

    候选 IP 列表变化时自动重新生成（marker 文件比对），保证 SAN 跟得上网络切换。
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    ips = lan_ips()
    marker = CERT_DIR / "ips.txt"
    if CERT_FILE.exists() and KEY_FILE.exists():
        try:
            if marker.read_text("utf-8").split() == ips:
                return CERT_FILE, KEY_FILE
        except Exception:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gpmerge-phone")])
    now = datetime.datetime.now(datetime.timezone.utc)
    san = x509.SubjectAlternativeName(
        [x509.DNSName("localhost"),
         x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
        + [x509.IPAddress(ipaddress.IPv4Address(ip)) for ip in ips])
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(san, critical=False)
            .sign(key, hashes.SHA256()))
    CERT_DIR.mkdir(exist_ok=True)
    KEY_FILE.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    marker.write_text("\n".join(ips), "utf-8")
    return CERT_FILE, KEY_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description="手机摄像头连拍服务")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-https", action="store_true",
                    help="仅本机调试用；局域网手机端没有 HTTPS 无法开摄像头")
    args = ap.parse_args()

    ips = lan_ips()
    scheme = "http" if args.no_https else "https"
    primary = ips[0] if ips else "127.0.0.1"
    url = f"{scheme}://{primary}:{args.port}"
    print("=" * 58)
    print("手机访问（同 Wi-Fi，从上往下试能打开的那个）:")
    for ip in ips:
        print(f"    {scheme}://{ip}:{args.port}")
    print(f"本机预览:      http://127.0.0.1:{args.port}")
    if not args.no_https:
        print("首次访问手机浏览器会提示证书风险：点「高级 -> 继续访问」")
    print("（Windows 防火墙首次弹窗请选\"允许访问\"）")
    print("=" * 58)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except ImportError:
        print("（pip install qrcode 后可在终端显示扫码图）")

    ssl = None if args.no_https else dict(zip(
        ("ssl_certfile", "ssl_keyfile"), ensure_cert()))
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
