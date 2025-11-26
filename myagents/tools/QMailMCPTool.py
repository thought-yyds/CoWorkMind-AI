"""
QQ 邮箱 FastMCP 服务（完整版）
功能：
 - 发送：plain / html / inline images / bulk / scheduled
 - 收件箱管理：list folders / list recent / search / read (text & html) / download attachments
 - 管理：mark read/unread / delete / move
 - MCP 增强：pydantic 校验、结构化返回、日志、错误类型、简单重试
依赖：fastmcp, pydantic, APScheduler
"""

import os
import imaplib
import smtplib
import threading
import time
import email
import logging
import mimetypes
from email import policy
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import decode_header, make_header
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, validator
from mcp.server.fastmcp import FastMCP


from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore




from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量


# -------------------------
# 基本配置 & logger
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qq-mail-mcp")

ATTACHMENTS_DIR = Path(os.getenv("QQ_MAIL_ATTACH_DIR", "./attachments"))
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 错误类 & 统一响应
# -------------------------
class QQMailConfigError(RuntimeError):
    pass

class QQMailOperationError(RuntimeError):
    pass

@dataclass
class MCPResponse:
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self):
        return {"ok": self.ok, "data": self.data, "error": self.error}

# -------------------------
# 工具函数
# -------------------------
def _get_credentials() -> tuple[str, str]:
    user = os.getenv("QQ_MAIL_ADDRESS")
    auth_code = os.getenv("QQ_MAIL_AUTH_CODE")
    if not user or not auth_code:
        raise QQMailConfigError("请设置环境变量 QQ_MAIL_ADDRESS 和 QQ_MAIL_AUTH_CODE")
    return user, auth_code

def _decode_header_value(raw_value: bytes | str | None) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8", errors="ignore")
    return str(make_header(decode_header(raw_value)))

def _connect_imap(timeout: int = 30) -> imaplib.IMAP4_SSL:
    user, auth_code = _get_credentials()
    host = os.getenv("QQ_MAIL_IMAP_SERVER", "imap.qq.com")
    port = int(os.getenv("QQ_MAIL_IMAP_PORT", "993"))
    client = imaplib.IMAP4_SSL(host, port, timeout=timeout)
    client.login(user, auth_code)
    return client

def _connect_smtp() -> smtplib.SMTP_SSL:
    user, auth_code = _get_credentials()
    host = os.getenv("QQ_MAIL_SMTP_SERVER", "smtp.qq.com")
    port = int(os.getenv("QQ_MAIL_SMTP_PORT", "465"))
    server = smtplib.SMTP_SSL(host, port)
    server.login(user, auth_code)
    return server

def _save_attachment(part: email.message.Message, target_dir: Path = ATTACHMENTS_DIR) -> str:
    filename = part.get_filename()
    if not filename:
        ext = mimetypes.guess_extension(part.get_content_type()) or ".bin"
        filename = f"attachment{int(time.time()*1000)}{ext}"
    safe_path = target_dir / filename
    payload = part.get_payload(decode=True)
    with open(safe_path, "wb") as fh:
        fh.write(payload or b"")
    return str(safe_path)

def _extract_text_and_html(msg: email.message.Message) -> Dict[str, str]:
    # Return {"text": "...", "html": "..."}
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get("Content-Disposition", "")
            # skip attachments
            if disp and disp.strip().startswith("attachment"):
                continue
            if ctype == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                text += (part.get_payload(decode=True) or b"").decode(charset, errors="ignore")
            elif ctype == "text/html":
                charset = part.get_content_charset() or "utf-8"
                html += (part.get_payload(decode=True) or b"").decode(charset, errors="ignore")
    else:
        ctype = msg.get_content_type()
        charset = msg.get_content_charset() or "utf-8"
        body = (msg.get_payload(decode=True) or b"").decode(charset, errors="ignore")
        if ctype == "text/html":
            html = body
        else:
            text = body
    return {"text": text.strip(), "html": html.strip()}

# -------------------------
# Pydantic schemas (参数校验)
# -------------------------
class SendMailSchema(BaseModel):
    recipient: str = Field(..., description="收件人邮箱")
    subject: str = Field(..., description="邮件主题")
    body: str = Field(..., description="邮件正文")
    attachments: Optional[List[str]] = None
    html: Optional[bool] = False
    inline_images: Optional[Dict[str, str]] = None  # cid => filepath

    @validator("recipient")
    def validate_recipient(cls, v):
        if "@" not in v:
            raise ValueError("recipient 必须是合法邮箱")
        return v

class BulkSendSchema(BaseModel):
    recipients: List[str]
    subject: str
    body: str
    html: Optional[bool] = False
    attachments: Optional[List[str]] = None

class ScheduleSendSchema(SendMailSchema):
    run_at_ts: int  # unix timestamp in seconds

class ListRecentSchema(BaseModel):
    limit: int = Field(5, ge=1, le=200)

class SearchSchema(BaseModel):
    keyword: str
    folder: Optional[str] = None
    limit: int = Field(20, ge=1, le=500)

class OperateUIDSchema(BaseModel):
    uid: str
    folder: Optional[str] = None

# -------------------------
# Scheduler
# -------------------------
scheduler = BackgroundScheduler(jobstores={"default": MemoryJobStore()})
scheduler.start()

def _schedule_send(job_id: str, send_args: dict, run_at_ts: int):
    def job_func():
        try:
            logger.info(f"[scheduler] 执行定时发送 job_id={job_id}")
            send_mail_impl(**send_args)
        except Exception as e:
            logger.exception("Scheduled send failed: %s", e)
    run_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_at_ts))
    scheduler.add_job(job_func, "date", run_date=run_date, id=job_id)
    logger.info(f"Scheduled job {job_id} at {run_date}")

# -------------------------
# 实际邮件发送实现（被工具和内部调用）
# -------------------------
def send_mail_impl(recipient: str, subject: str, body: str, attachments: Optional[List[str]] = None,
                   html: bool = False, inline_images: Optional[Dict[str,str]] = None, retry: int = 1) -> str:
    sender, _ = _get_credentials()
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Body
    if html:
        msg.attach(MIMEText(body, "html", "utf-8"))
    else:
        msg.attach(MIMEText(body, "plain", "utf-8"))

    # Inline images (cid)
    if inline_images:
        for cid, path in inline_images.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Inline image not found: {path}")
            with open(path, "rb") as fh:
                part = MIMEApplication(fh.read())
                part.add_header("Content-ID", f"<{cid}>")
                part.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
                msg.attach(part)

    # Attachments
    attachments = attachments or []
    for path in attachments:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attachment not found: {path}")
        with open(path, "rb") as fh:
            part = MIMEApplication(fh.read())
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
            msg.attach(part)

    last_exc = None
    for attempt in range(retry):
        try:
            with _connect_smtp() as server:
                server.sendmail(sender, [recipient], msg.as_string())
            return f"Mail sent to {recipient}."
        except Exception as e:
            last_exc = e
            logger.exception("SMTP send failed, attempt %d/%d", attempt + 1, retry)
            time.sleep(1 + attempt * 2)
    raise QQMailOperationError(f"Send failed after {retry} attempts: {last_exc}")

# -------------------------
# FastMCP 服务绑定
# -------------------------
app = FastMCP("qq-mail-mcp")

# -------------------------
# A: 发送相关工具
# -------------------------
@app.tool(name="send_mail")
def send_mail(recipient: str, subject: str, body: str, attachments: Optional[List[str]] = None,
              html: Optional[bool] = False, inline_images: Optional[Dict[str,str]] = None,
              retry: int = 2) -> dict:
    """发送单封邮件（plain 或 html），支持内嵌图片和附件"""
    try:
        params = SendMailSchema(
            recipient=recipient, subject=subject, body=body,
            attachments=attachments, html=html, inline_images=inline_images
        )
        result = send_mail_impl(**params.dict(), retry=retry)
        return MCPResponse(ok=True, data={"result": result}).to_dict()
    except Exception as e:
        logger.exception("send_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="send_html")
def send_html(recipient: str, subject: str, html_body: str, attachments: Optional[List[str]] = None,
              inline_images: Optional[Dict[str,str]] = None, retry: int = 2) -> dict:
    """专门发送 HTML 邮件（包装）"""
    return send_mail(recipient=recipient, subject=subject, body=html_body,
                     attachments=attachments, html=True, inline_images=inline_images, retry=retry)

@app.tool(name="send_with_inline_images")
def send_with_inline_images(recipient: str, subject: str, html_body: str, inline_images: Dict[str,str],
                            attachments: Optional[List[str]] = None, retry: int = 2) -> dict:
    """发送 HTML + 内嵌图片，inline_images: {"logo":"./logo.png"}，在 html 中用 <img src='cid:logo' />"""
    return send_html(recipient=recipient, subject=subject, html_body=html_body,
                     attachments=attachments, inline_images=inline_images, retry=retry)

@app.tool(name="bulk_send")
def bulk_send(recipients: List[str], subject: str, body: str, html: Optional[bool] = False,
              attachments: Optional[List[str]] = None, batch_size: int = 10) -> dict:
    """
    批量发送（简单实现：串行发送，可根据需求改为并行或分批延迟）
    返回：{sent: N, failed: [{recipient, error}], total: M}
    """
    try:
        params = BulkSendSchema(recipients=recipients, subject=subject, body=body, html=html, attachments=attachments)
    except Exception as e:
        return MCPResponse(ok=False, error=str(e)).to_dict()

    sent = 0
    failed = []
    for r in params.recipients:
        try:
            send_mail_impl(recipient=r, subject=params.subject, body=params.body,
                           attachments=params.attachments, html=params.html, retry=1)
            sent += 1
        except Exception as e:
            logger.exception("bulk_send failed for %s", r)
            failed.append({"recipient": r, "error": str(e)})
    return MCPResponse(ok=True, data={"sent": sent, "failed": failed, "total": len(params.recipients)}).to_dict()

@app.tool(name="schedule_send")
def schedule_send(recipient: str, subject: str, body: str, run_at_ts: int,
                  attachments: Optional[List[str]] = None, html: Optional[bool] = False,
                  inline_images: Optional[Dict[str,str]] = None) -> dict:
    """
    本地定时发送（基于 APScheduler）
    run_at_ts: unix 时间戳（秒）
    """
    try:
        params = ScheduleSendSchema(recipient=recipient, subject=subject, body=body,
                                    attachments=attachments, html=html, inline_images=inline_images,
                                    run_at_ts=run_at_ts)
    except Exception as e:
        return MCPResponse(ok=False, error=str(e)).to_dict()

    job_id = f"send_{int(time.time()*1000)}"
    _schedule_send(job_id, params.dict(exclude={"run_at_ts"}), params.run_at_ts)
    return MCPResponse(ok=True, data={"job_id": job_id, "run_at": params.run_at_ts}).to_dict()

# -------------------------
# B: 收件管理相关工具
# -------------------------
@app.tool(name="list_folders")
def list_folders() -> dict:
    """列出 IMAP 文件夹（mailboxes）"""
    try:
        with _connect_imap() as client:
            typ, data = client.list()
            if typ != "OK":
                raise QQMailOperationError("LIST failed")
            folders = []
            for line in data:
                if not line:
                    continue
                decoded = line.decode()
                # 常见格式： b'(\\HasNoChildren) "." "INBOX"' -> parse name between quotes
                try:
                    parts = decoded.split(' "/" ')
                    # fallback: take last quoted
                    name = decoded.split()[-1].strip('"')
                except Exception:
                    name = decoded
                folders.append(name.strip('"'))
            return MCPResponse(ok=True, data={"folders": folders}).to_dict()
    except Exception as e:
        logger.exception("list_folders error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="list_recent_mail")
def list_recent_mail(limit: int = 5, folder: Optional[str] = "INBOX") -> dict:
    """列出最近邮件（返回 uid, subject, from, date, snippet）"""
    try:
        params = ListRecentSchema(limit=limit)
        with _connect_imap() as client:
            client.select(folder)
            typ, data = client.search(None, "ALL")
            if typ != "OK":
                return MCPResponse(ok=True, data={"mails": []}).to_dict()
            ids = data[0].split()
            selected = ids[-params.limit:]
            summaries = []
            for uid in reversed(selected):
                typ, msg_data = client.fetch(uid, "(RFC822)")
                if typ != "OK" or not msg_data:
                    continue
                msg = email.message_from_bytes(msg_data[0][1], policy=policy.default)
                text_html = _extract_text_and_html(msg)
                snippet = (text_html["text"] or text_html["html"] or "")[:200]
                summaries.append({
                    "uid": uid.decode(),
                    "subject": _decode_header_value(msg.get("Subject")),
                    "from": _decode_header_value(msg.get("From")),
                    "date": msg.get("Date"),
                    "snippet": snippet
                })
            return MCPResponse(ok=True, data={"mails": summaries}).to_dict()
    except Exception as e:
        logger.exception("list_recent_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="read_mail")
def read_mail(uid: str, folder: Optional[str] = "INBOX", download_attachments: Optional[bool] = False) -> dict:
    """读取指定 UID 邮件（返回 text / html / attachments 列表）"""
    try:
        params = OperateUIDSchema(uid=uid, folder=folder)
        with _connect_imap() as client:
            client.select(params.folder)
            typ, msg_data = client.fetch(params.uid.encode(), "(RFC822)")
            if typ != "OK" or not msg_data:
                raise QQMailOperationError(f"邮件 {params.uid} 未找到")
            msg = email.message_from_bytes(msg_data[0][1], policy=policy.default)
            text_html = _extract_text_and_html(msg)
            attachments = []
            if msg.is_multipart():
                for part in msg.walk():
                    disp = part.get("Content-Disposition", "")
                    if disp and disp.strip().startswith("attachment"):
                        saved = _save_attachment(part)
                        attachments.append(saved)
            # optional: download inline images as well
            if download_attachments:
                # also attempt to save inline parts with content-id
                if msg.is_multipart():
                    for part in msg.walk():
                        cid = part.get("Content-ID")
                        disp = part.get("Content-Disposition", "")
                        if cid and (not disp or not disp.strip().startswith("attachment")):
                            saved = _save_attachment(part)
                            attachments.append(saved)
            return MCPResponse(ok=True, data={
                "uid": params.uid,
                "subject": _decode_header_value(msg.get("Subject")),
                "from": _decode_header_value(msg.get("From")),
                "to": _decode_header_value(msg.get("To")),
                "date": msg.get("Date"),
                "text": text_html["text"],
                "html": text_html["html"],
                "attachments": attachments
            }).to_dict()
    except Exception as e:
        logger.exception("read_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="download_attachments")
def download_attachments(uid: str, folder: Optional[str] = "INBOX") -> dict:
    """下载指定邮件的所有附件并返回路径列表"""
    try:
        params = OperateUIDSchema(uid=uid, folder=folder)
        with _connect_imap() as client:
            client.select(params.folder)
            typ, msg_data = client.fetch(params.uid.encode(), "(RFC822)")
            if typ != "OK" or not msg_data:
                raise QQMailOperationError("fetch failed")
            msg = email.message_from_bytes(msg_data[0][1], policy=policy.default)
            saved = []
            if msg.is_multipart():
                for part in msg.walk():
                    disp = part.get("Content-Disposition", "")
                    if disp and disp.strip().startswith("attachment"):
                        saved_path = _save_attachment(part)
                        saved.append(saved_path)
            return MCPResponse(ok=True, data={"saved": saved}).to_dict()
    except Exception as e:
        logger.exception("download_attachments error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="search_mail")
def search_mail(keyword: str, folder: Optional[str] = "INBOX", limit: int = 50) -> dict:
    """
    按关键词搜索（IMAP BODY/HEADER 搜索能力有限）
    注意：IMAP 搜索在不同服务器表现不同，复杂检索建议下载后本地索引
    """
    try:
        params = SearchSchema(keyword=keyword, folder=folder, limit=limit)
        with _connect_imap() as client:
            client.select(params.folder)
            # 使用 BODY 搜索可能匹配正文，也可以组合 FROM/TO/SUBJECT
            typ, data = client.search(None, f'BODY "{params.keyword}"')
            if typ != "OK":
                return MCPResponse(ok=True, data={"mails": []}).to_dict()
            ids = data[0].split()[-params.limit:]
            results = []
            for uid in reversed(ids):
                typ, msg_data = client.fetch(uid, "(RFC822)")
                if typ != "OK" or not msg_data:
                    continue
                msg = email.message_from_bytes(msg_data[0][1], policy=policy.default)
                results.append({
                    "uid": uid.decode(),
                    "subject": _decode_header_value(msg.get("Subject")),
                    "from": _decode_header_value(msg.get("From")),
                    "date": msg.get("Date")
                })
            return MCPResponse(ok=True, data={"mails": results}).to_dict()
    except Exception as e:
        logger.exception("search_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="mark_read")
def mark_read(uid: str, folder: Optional[str] = "INBOX") -> dict:
    try:
        params = OperateUIDSchema(uid=uid, folder=folder)
        with _connect_imap() as client:
            client.select(params.folder)
            # set \Seen flag
            client.store(params.uid.encode(), "+FLAGS", "\\Seen")
        return MCPResponse(ok=True, data={"uid": params.uid, "marked": "read"}).to_dict()
    except Exception as e:
        logger.exception("mark_read error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="mark_unread")
def mark_unread(uid: str, folder: Optional[str] = "INBOX") -> dict:
    try:
        params = OperateUIDSchema(uid=uid, folder=folder)
        with _connect_imap() as client:
            client.select(params.folder)
            client.store(params.uid.encode(), "-FLAGS", "\\Seen")
        return MCPResponse(ok=True, data={"uid": params.uid, "marked": "unread"}).to_dict()
    except Exception as e:
        logger.exception("mark_unread error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="delete_mail")
def delete_mail(uid: str, folder: Optional[str] = "INBOX") -> dict:
    try:
        params = OperateUIDSchema(uid=uid, folder=folder)
        with _connect_imap() as client:
            client.select(params.folder)
            # 标记删除（\Deleted），并 EXPUNGE
            client.store(params.uid.encode(), "+FLAGS", "\\Deleted")
            client.expunge()
        return MCPResponse(ok=True, data={"uid": params.uid, "deleted": True}).to_dict()
    except Exception as e:
        logger.exception("delete_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

@app.tool(name="move_mail")
def move_mail(uid: str, target_folder: str, src_folder: Optional[str] = "INBOX") -> dict:
    """
    将邮件移动到 target_folder（使用 COPY + STORE + EXPUNGE 的通用方法）
    """
    try:
        params = OperateUIDSchema(uid=uid, folder=src_folder)
        with _connect_imap() as client:
            client.select(params.folder)
            # COPY 到目标文件夹
            typ, resp = client.copy(params.uid.encode(), target_folder)
            if typ != "OK":
                raise QQMailOperationError("copy failed: " + str(resp))
            # 标记原始为 Deleted 并 expunge
            client.store(params.uid.encode(), "+FLAGS", "\\Deleted")
            client.expunge()
        return MCPResponse(ok=True, data={"uid": params.uid, "moved_to": target_folder}).to_dict()
    except Exception as e:
        logger.exception("move_mail error")
        return MCPResponse(ok=False, error=str(e)).to_dict()

# -------------------------
# 启动（当做脚本直接运行）
# -------------------------
if __name__ == "__main__":
    logger.info("Starting QQ Mail MCP service...")
    try:
        # 测试凭证是否存在
        _get_credentials()
    except Exception as e:
        logger.error("配置错误: %s", e)
        raise SystemExit(1)
    app.run()
