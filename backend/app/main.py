from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    and_,
    create_engine,
    event,
    func,
    or_,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./support_panel.db")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"


class Base(DeclarativeBase):
    pass


class RoleEnum(str, Enum):
    junior = "junior"
    senior = "senior"
    admin = "admin"


class SenderType(str, Enum):
    user = "user"
    staff = "staff"
    system = "system"


class AttachmentType(str, Enum):
    photo = "photo"
    video = "video"


class AuditAction(str, Enum):
    login = "login"
    ticket_reply = "ticket_reply"
    ticket_close = "ticket_close"
    ticket_reopen = "ticket_reopen"
    ticket_note_update = "ticket_note_update"
    ticket_refund_flag_update = "ticket_refund_flag_update"
    ban_add = "ban_add"
    ban_remove = "ban_remove"
    template_create = "template_create"
    template_update = "template_update"
    template_delete = "template_delete"
    quickbtn_create = "quickbtn_create"
    quickbtn_update = "quickbtn_update"
    quickbtn_delete = "quickbtn_delete"
    import_archive = "import_archive"


class Staff(Base):
    __tablename__ = "staff"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[RoleEnum] = mapped_column(SqlEnum(RoleEnum), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    telegram_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    registered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telegram_chat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    telegram_topic_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_telegram_id: Mapped[int] = mapped_column(ForeignKey("users.telegram_id"), nullable=False)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    internal_note: Mapped[str] = mapped_column(Text, default="")
    admin_flag_refund: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    source_txt_path: Mapped[str | None] = mapped_column(String, nullable=True)


class TicketParticipant(Base):
    __tablename__ = "ticket_participants"
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False)
    staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id"), nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    __table_args__ = (PrimaryKeyConstraint("ticket_id", "staff_id"),)


class TicketMessage(Base):
    __tablename__ = "ticket_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False)
    sender_type: Mapped[SenderType] = mapped_column(SqlEnum(SenderType), nullable=False)
    sender_staff_id: Mapped[int | None] = mapped_column(ForeignKey("staff.id"), nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class TicketAttachment(Base):
    __tablename__ = "ticket_attachments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("ticket_messages.id"), nullable=False)
    type: Mapped[AttachmentType] = mapped_column(SqlEnum(AttachmentType), nullable=False)
    telegram_file_id: Mapped[str] = mapped_column(String, nullable=False)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class CannedTemplate(Base):
    __tablename__ = "canned_templates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QuickButton(Base):
    __tablename__ = "quick_buttons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BanList(Base):
    __tablename__ = "banlist"
    telegram_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revoked_by_staff_id: Mapped[int | None] = mapped_column(ForeignKey("staff.id"), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    staff_id: Mapped[int] = mapped_column(ForeignKey("staff.id"), nullable=False)
    action: Mapped[AuditAction] = mapped_column(SqlEnum(AuditAction), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String, nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String, nullable=True)
    meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, _):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


class WSManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, event_type: str, payload: dict[str, Any]):
        dead = []
        for conn in self.connections:
            try:
                await conn.send_json({"type": event_type, "payload": payload})
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


manager = WSManager()
app = FastAPI(title="Hyper Collision Support API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TelegramAuthPayload(BaseModel):
    id: int
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None
    auth_date: int
    hash: str


class TicketReplyPayload(BaseModel):
    text: str
    attachments: list[dict[str, Any]] = []


class TicketPatchPayload(BaseModel):
    internal_note: str | None = None
    admin_flag_refund: bool | None = None


class ParticipantPayload(BaseModel):
    staff_id: int


class BanPayload(BaseModel):
    telegram_id: int
    reason: str


class UnbanPayload(BaseModel):
    telegram_id: int


class ImportPayload(BaseModel):
    path: str | None = None
    files: list[str] | None = None


class TemplatePayload(BaseModel):
    title: str
    body: str


class QuickButtonPayload(BaseModel):
    title: str
    payload: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(data: dict[str, Any], expires_delta: timedelta = timedelta(hours=12)) -> str:
    to_encode = data.copy()
    to_encode.update({"exp": datetime.now(timezone.utc) + expires_delta})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def get_current_staff(authorization: str = Header(default=""), db: Session = Depends(get_db)) -> Staff:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Требуется токен")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Неверный токен") from exc
    staff_id = payload.get("staff_id")
    staff = db.get(Staff, staff_id)
    if not staff or not staff.is_active:
        raise HTTPException(status_code=403, detail="Сотрудник недоступен")
    return staff


def audit(db: Session, staff_id: int, action: AuditAction, entity_type: str | None = None, entity_id: str | None = None, meta: dict[str, Any] | None = None):
    db.add(AuditLog(staff_id=staff_id, action=action, entity_type=entity_type, entity_id=entity_id, meta_json=json.dumps(meta, ensure_ascii=False) if meta else None))


def assert_not_archived(ticket: Ticket):
    if ticket.is_archived:
        raise HTTPException(status_code=400, detail="Архивный тикет доступен только для чтения")


async def send_to_telegram(ticket: Ticket, text: str):
    if not BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": ticket.telegram_chat_id, "message_thread_id": ticket.telegram_topic_id, "text": text}
    async with httpx.AsyncClient(timeout=10) as client:
        for _ in range(3):
            try:
                await client.post(url, json=payload)
                return
            except Exception:
                continue


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)


@app.post("/auth/telegram")
def auth_telegram(payload: TelegramAuthPayload, db: Session = Depends(get_db)):
    data_check = {k: v for k, v in payload.model_dump().items() if k != "hash" and v is not None}
    check_string = "\n".join(f"{k}={data_check[k]}" for k in sorted(data_check))
    secret = hashlib.sha256(BOT_TOKEN.encode()).digest()
    digest = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    if BOT_TOKEN and digest != payload.hash:
        raise HTTPException(status_code=401, detail="Подпись Telegram не прошла проверку")
    staff = db.execute(select(Staff).where(Staff.telegram_id == payload.id, Staff.is_active.is_(True))).scalar_one_or_none()
    if not staff:
        raise HTTPException(status_code=403, detail="Доступ только для сотрудников")
    token = create_access_token({"staff_id": staff.id, "telegram_id": staff.telegram_id, "role": staff.role.value})
    audit(db, staff.id, AuditAction.login)
    db.commit()
    return {"access_token": token, "token_type": "bearer"}


@app.get("/tickets")
def list_tickets(state: str = "all", archived: str = "all", limit: int = 50, offset: int = 0, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    stmt = select(Ticket)
    if state == "open":
        stmt = stmt.where(Ticket.closed_at.is_(None))
    elif state == "closed":
        stmt = stmt.where(Ticket.closed_at.is_not(None))
    if archived == "0":
        stmt = stmt.where(Ticket.is_archived.is_(False))
    elif archived == "1":
        stmt = stmt.where(Ticket.is_archived.is_(True))
    rows = db.execute(stmt.order_by(Ticket.last_message_at.desc()).limit(limit).offset(offset)).scalars().all()
    return rows


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    messages = db.execute(select(TicketMessage).where(TicketMessage.ticket_id == ticket_id).order_by(TicketMessage.created_at)).scalars().all()
    participants = db.execute(select(TicketParticipant).where(TicketParticipant.ticket_id == ticket_id)).scalars().all()
    return {"ticket": ticket, "messages": messages, "participants": participants}


@app.post("/tickets/{ticket_id}/reply")
async def reply_ticket(ticket_id: int, payload: TicketReplyPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    await send_to_telegram(ticket, payload.text)
    message = TicketMessage(ticket_id=ticket.id, sender_type=SenderType.staff, sender_staff_id=staff.id, text=payload.text)
    db.add(message)
    db.flush()
    for att in payload.attachments:
        db.add(TicketAttachment(message_id=message.id, type=att["type"], telegram_file_id=att["telegram_file_id"], meta_json=json.dumps(att.get("meta_json")) if att.get("meta_json") else None))
    ticket.last_message_at = datetime.utcnow()
    audit(db, staff.id, AuditAction.ticket_reply, "ticket", str(ticket_id))
    db.commit()
    await manager.broadcast("MESSAGE_CREATED", {"ticket_id": ticket_id, "message_id": message.id, "text": message.text})
    return {"ok": True}


@app.post("/tickets/{ticket_id}/close")
async def close_ticket(ticket_id: int, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    ticket.closed_at = datetime.utcnow()
    audit(db, staff.id, AuditAction.ticket_close, "ticket", str(ticket_id))
    db.commit()
    await manager.broadcast("TICKET_UPDATED", {"id": ticket.id, "closed_at": ticket.closed_at.isoformat()})
    return {"ok": True}


@app.post("/tickets/{ticket_id}/reopen")
async def reopen_ticket(ticket_id: int, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    ticket.closed_at = None
    audit(db, staff.id, AuditAction.ticket_reopen, "ticket", str(ticket_id))
    db.commit()
    await manager.broadcast("TICKET_UPDATED", {"id": ticket.id, "closed_at": None})
    return {"ok": True}


@app.patch("/tickets/{ticket_id}")
async def patch_ticket(ticket_id: int, payload: TicketPatchPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    if payload.internal_note is not None:
        ticket.internal_note = payload.internal_note
        audit(db, staff.id, AuditAction.ticket_note_update, "ticket", str(ticket_id))
    if payload.admin_flag_refund is not None:
        ticket.admin_flag_refund = payload.admin_flag_refund
        audit(db, staff.id, AuditAction.ticket_refund_flag_update, "ticket", str(ticket_id))
    db.commit()
    await manager.broadcast("TICKET_UPDATED", {"id": ticket.id, "internal_note": ticket.internal_note, "admin_flag_refund": ticket.admin_flag_refund})
    return {"ok": True}


@app.post("/tickets/{ticket_id}/participants/add")
def add_participant(ticket_id: int, payload: ParticipantPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    item = TicketParticipant(ticket_id=ticket_id, staff_id=payload.staff_id)
    db.merge(item)
    audit(db, staff.id, AuditAction.ticket_note_update, "ticket", str(ticket_id), {"participant_add": payload.staff_id})
    db.commit()
    return {"ok": True}


@app.post("/tickets/{ticket_id}/participants/remove")
def remove_participant(ticket_id: int, payload: ParticipantPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Тикет не найден")
    assert_not_archived(ticket)
    db.query(TicketParticipant).filter_by(ticket_id=ticket_id, staff_id=payload.staff_id).delete()
    audit(db, staff.id, AuditAction.ticket_note_update, "ticket", str(ticket_id), {"participant_remove": payload.staff_id})
    db.commit()
    return {"ok": True}


@app.get("/tickets/search")
def search_tickets(q: str, include_archived: bool = True, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    msg_ticket_ids = [row[0] for row in db.execute(select(TicketMessage.ticket_id).where(TicketMessage.text.ilike(f"%{q}%")).distinct()).all()]
    stmt = select(Ticket).where(
        or_(
            Ticket.id == (int(q) if q.isdigit() else -1),
            Ticket.user_telegram_id == (int(q) if q.isdigit() else -1),
            Ticket.username.ilike(f"%{q}%"),
            Ticket.id.in_(msg_ticket_ids or [-1]),
        )
    )
    if not include_archived:
        stmt = stmt.where(Ticket.is_archived.is_(False))
    return db.execute(stmt.order_by(Ticket.last_message_at.desc())).scalars().all()


def _crud_routes(path: str, model, payload_cls, create_action: AuditAction, update_action: AuditAction, delete_action: AuditAction, entity_type: str):
    @app.get(path)
    def _list(_: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
        return db.execute(select(model).order_by(model.id.desc())).scalars().all()

    @app.post(path)
    def _create(payload: payload_cls, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
        item = model(**payload.model_dump(), created_by_staff_id=staff.id)
        db.add(item)
        audit(db, staff.id, create_action, entity_type)
        db.commit()
        return item

    @app.put(f"{path}/{{item_id}}")
    def _update(item_id: int, payload: payload_cls, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
        item = db.get(model, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Не найдено")
        for k, v in payload.model_dump().items():
            setattr(item, k, v)
        item.updated_at = datetime.utcnow()
        audit(db, staff.id, update_action, entity_type, str(item_id))
        db.commit()
        return item

    @app.delete(f"{path}/{{item_id}}")
    def _delete(item_id: int, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
        item = db.get(model, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Не найдено")
        db.delete(item)
        audit(db, staff.id, delete_action, entity_type, str(item_id))
        db.commit()
        return {"ok": True}


_crud_routes("/templates", CannedTemplate, TemplatePayload, AuditAction.template_create, AuditAction.template_update, AuditAction.template_delete, "template")
_crud_routes("/quick-buttons", QuickButton, QuickButtonPayload, AuditAction.quickbtn_create, AuditAction.quickbtn_update, AuditAction.quickbtn_delete, "quick_button")


@app.get("/users/{telegram_id}")
def get_user(telegram_id: int, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    user = db.get(User, telegram_id)
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    tickets = db.execute(select(Ticket).where(Ticket.user_telegram_id == telegram_id).order_by(Ticket.created_at.desc())).scalars().all()
    return {"user": user, "tickets": tickets}


@app.get("/users")
def users(search: str | None = None, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    stmt = select(User)
    if search:
        stmt = stmt.where(or_(User.username.ilike(f"%{search}%"), User.telegram_id == (int(search) if search.isdigit() else -1)))
    return db.execute(stmt.limit(100)).scalars().all()


@app.get("/banlist")
def banlist(_: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    return db.execute(select(BanList).order_by(BanList.created_at.desc())).scalars().all()


@app.post("/banlist/ban")
async def ban(payload: BanPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    item = BanList(telegram_id=payload.telegram_id, reason=payload.reason, created_by_staff_id=staff.id, is_active=True)
    db.merge(item)
    audit(db, staff.id, AuditAction.ban_add, "ban", str(payload.telegram_id))
    db.commit()
    await manager.broadcast("BAN_UPDATED", {"telegram_id": payload.telegram_id, "is_active": True})
    return {"ok": True}


@app.post("/banlist/unban")
async def unban(payload: UnbanPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    item = db.get(BanList, payload.telegram_id)
    if not item:
        raise HTTPException(status_code=404, detail="Не найден в банлисте")
    item.is_active = False
    item.revoked_at = datetime.utcnow()
    item.revoked_by_staff_id = staff.id
    audit(db, staff.id, AuditAction.ban_remove, "ban", str(payload.telegram_id))
    db.commit()
    await manager.broadcast("BAN_UPDATED", {"telegram_id": payload.telegram_id, "is_active": False})
    return {"ok": True}


@app.get("/audit")
def get_audit(from_: datetime | None = Query(None, alias="from"), to: datetime | None = None, staff_id: int | None = None, _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    stmt = select(AuditLog)
    if from_:
        stmt = stmt.where(AuditLog.created_at >= from_)
    if to:
        stmt = stmt.where(AuditLog.created_at <= to)
    if staff_id:
        stmt = stmt.where(AuditLog.staff_id == staff_id)
    return db.execute(stmt.order_by(AuditLog.created_at.desc())).scalars().all()


@app.get("/analytics/dashboard")
def analytics_dashboard(from_: datetime = Query(alias="from"), to: datetime = Query(alias="to"), _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    tickets = db.execute(select(Ticket).where(and_(Ticket.created_at >= from_, Ticket.created_at <= to))).scalars().all()
    total = len(tickets)
    open_tickets = sum(1 for t in tickets if t.closed_at is None and not t.is_archived)
    first_response_list = []
    close_list = []
    for t in tickets:
        first = db.execute(select(TicketMessage.created_at).where(TicketMessage.ticket_id == t.id, TicketMessage.sender_type == SenderType.staff).order_by(TicketMessage.created_at)).scalar_one_or_none()
        if first:
            first_response_list.append((first - t.created_at).total_seconds())
        if t.closed_at:
            close_list.append((t.closed_at - t.created_at).total_seconds())
    return {
        "total_tickets": total,
        "open_tickets": open_tickets,
        "avg_time_to_first_response": (sum(first_response_list) / len(first_response_list)) if first_response_list else None,
        "avg_time_to_close": (sum(close_list) / len(close_list)) if close_list else None,
    }


@app.get("/analytics/export")
def analytics_export(from_: datetime = Query(alias="from"), to: datetime = Query(alias="to"), format: str = "csv", _: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    tickets = db.execute(select(Ticket).where(and_(Ticket.created_at >= from_, Ticket.created_at <= to))).scalars().all()
    rows = []
    for t in tickets:
        first = db.execute(select(TicketMessage.created_at).where(TicketMessage.ticket_id == t.id, TicketMessage.sender_type == SenderType.staff).order_by(TicketMessage.created_at)).scalar_one_or_none()
        rows.append({
            "ticket_id": t.id,
            "telegram_id": t.user_telegram_id,
            "username": t.username,
            "created_at": t.created_at,
            "first_staff_response_at": first,
            "closed_at": t.closed_at,
            "time_to_first_response": (first - t.created_at).total_seconds() if first else None,
            "time_to_close": (t.closed_at - t.created_at).total_seconds() if t.closed_at else None,
            "refund_flag": t.admin_flag_refund,
        })
    if format == "xlsx":
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        headers = list(rows[0].keys()) if rows else ["ticket_id", "telegram_id", "username", "created_at", "first_staff_response_at", "closed_at", "time_to_first_response", "time_to_close", "refund_flag"]
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h) for h in headers])
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return StreamingResponse(bio, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=["ticket_id", "telegram_id", "username", "created_at", "first_staff_response_at", "closed_at", "time_to_first_response", "time_to_close", "refund_flag"])
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(iter([sio.getvalue()]), media_type="text/csv")


@app.post("/admin/import-archive")
def import_archive(payload: ImportPayload, staff: Staff = Depends(get_current_staff), db: Session = Depends(get_db)):
    files = payload.files or []
    if payload.path:
        files.extend(str(p) for p in Path(payload.path).glob("ticket_*.txt"))
    imported, skipped, errors = 0, 0, []
    for file_path in files:
        try:
            m = re.search(r"ticket_(\d+)\.txt", Path(file_path).name)
            if not m:
                skipped += 1
                continue
            ticket_id = int(m.group(1))
            if db.get(Ticket, ticket_id):
                skipped += 1
                continue
            lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
            now = datetime.utcnow()
            created = now
            last = now
            ticket = Ticket(id=ticket_id, telegram_chat_id=0, telegram_topic_id=0, user_telegram_id=0, is_archived=True, source_txt_path=str(Path(file_path).resolve()), created_at=created, last_message_at=last, closed_at=last)
            db.add(ticket)
            db.flush()
            for ln in lines:
                sm = re.match(r"\[(.*?)\]\s+(\w+)\s+(\d+):\s*(.*)", ln)
                if not sm:
                    continue
                actor, sender_tag, uid, text = sm.groups()
                sender_type = SenderType.user if sender_tag.upper() == "USER" else SenderType.staff
                if sender_type == SenderType.staff:
                    staff_obj = db.execute(select(Staff).where(Staff.display_name == actor)).scalar_one_or_none()
                    sender_staff_id = staff_obj.id if staff_obj else None
                    if sender_staff_id is None:
                        sender_type = SenderType.system
                else:
                    sender_staff_id = None
                db.add(TicketMessage(ticket_id=ticket_id, sender_type=sender_type, sender_staff_id=sender_staff_id, text=text, created_at=now))
            imported += 1
        except Exception as exc:
            errors.append({"file": file_path, "error": str(exc)})
    audit(db, staff.id, AuditAction.import_archive, "archive", None, {"imported": imported, "skipped": skipped, "errors": len(errors)})
    db.commit()
    return {"imported": imported, "skipped": skipped, "errors": errors}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = Query(default="")):
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError:
        await websocket.close(code=4001)
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
