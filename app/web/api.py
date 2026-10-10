"""JSON API blueprint for Admin v2."""
from __future__ import annotations

import logging
import secrets
import sqlite3

from flask import Blueprint, g, jsonify, request, make_response

from app.web import queries
from app.web.contracts import CSRF_COOKIE, CSRF_FORM_FIELD, CSRF_HEADER
from core.bus.queue import JobStore

log = logging.getLogger("kizaru.web")
api_bp = Blueprint("api", __name__)


def _db():
    return g.db_path


def _csrf_for_request() -> str:
    existing = request.cookies.get(CSRF_COOKIE)
    if existing:
        g._csrf_token = existing
        return existing
    token = getattr(g, "_csrf_token", None)
    if not token:
        token = secrets.token_urlsafe(32)
        g._csrf_token = token
    return token


def _ensure_csrf(resp):
    token = _csrf_for_request()
    if not request.cookies.get(CSRF_COOKIE):
        resp.set_cookie(
            CSRF_COOKIE,
            token,
            httponly=False,
            samesite="Strict",
            path="/",
        )
    return token


def _check_csrf() -> bool:
    cookie = request.cookies.get(CSRF_COOKIE) or ""
    header = request.headers.get(CSRF_HEADER) or ""
    form = ""
    if request.form:
        form = request.form.get(CSRF_FORM_FIELD) or ""
    if request.is_json and isinstance(request.json, dict):
        form = form or (request.json.get(CSRF_FORM_FIELD) or "")
    supplied = header or form
    return bool(cookie) and bool(supplied) and secrets.compare_digest(cookie, supplied)


@api_bp.after_request
def _csrf_cookie(resp):
    if request.path.startswith("/api/"):
        _ensure_csrf(resp)
        resp.headers["Cache-Control"] = "no-store"
    return resp


@api_bp.get("/api/home")
def home():
    try:
        return jsonify(queries.api_home(_db()))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/queue")
def queue():
    try:
        return jsonify(queries.api_queue_v2(_db()))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/jobs")
def jobs_list():
    try:
        page = request.args.get("page", 1, type=int) or 1
        return jsonify(
            queries.api_jobs_page(
                _db(),
                status=(request.args.get("status") or "").strip() or None,
                kind=(request.args.get("kind") or "").strip() or None,
                user=(request.args.get("user") or "").strip() or None,
                page=page,
            )
        )
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/jobs/<int:job_id>")
def job_detail(job_id: int):
    try:
        data = queries.api_job_detail(_db(), job_id)
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@api_bp.post("/api/jobs/<int:job_id>/cancel")
def job_cancel(job_id: int):
    if not _check_csrf():
        return jsonify({"error": "csrf"}), 400
    try:
        detail = queries.api_job_detail(_db(), job_id)
        if detail is None:
            return jsonify({"error": "not found"}), 404
        store = JobStore(_db(), recover_running=False)
        ok = store.cancel_job(job_id)
        if not ok:
            return jsonify({"error": "not cancellable"}), 400
        return jsonify({"ok": True, "id": job_id, "status": "cancelled"})
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/users")
def users_list():
    hide = request.args.get("hide_cli", "") in ("1", "true", "yes")
    try:
        return jsonify(queries.api_users_list(_db(), hide_cli=hide))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/users/<path:user_id>")
def user_detail(user_id: str):
    try:
        data = queries.api_user_detail(_db(), user_id)
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@api_bp.post("/api/users/<path:user_id>/profile")
def user_profile(user_id: str):
    if not _check_csrf():
        return jsonify({"error": "csrf"}), 400
    try:
        data = queries.api_user_profile(_db(), user_id)
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@api_bp.post("/api/users/<path:user_id>/delete")
def user_delete(user_id: str):
    if not _check_csrf():
        return jsonify({"error": "csrf"}), 400
    try:
        result = queries.api_user_delete(_db(), user_id)
        log.info("user delete id_mask=%s deleted=%s", result.get("id_mask"), result.get("deleted"))
        return jsonify(result)
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/services")
def services():
    try:
        return jsonify(queries.api_services(_db()))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/workers")
def workers():
    try:
        return jsonify(queries.api_workers(_db()))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/models")
def models():
    try:
        return jsonify(queries.api_models(_db()))
    except Exception:
        log.exception("admin models error")
        return jsonify({"error": "models error"}), 500


@api_bp.get("/api/stats/timeseries")
def timeseries():
    try:
        return jsonify(
            queries.api_timeseries(
                _db(),
                metric=(request.args.get("metric") or "jobs_created").strip(),
                range_key=(request.args.get("range") or "24h").strip(),
            )
        )
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/logs")
def logs():
    try:
        return jsonify(
            queries.api_logs(
                service=(request.args.get("service") or "").strip() or None,
                level=(request.args.get("level") or "").strip() or None,
            )
        )
    except Exception:
        log.exception("admin logs error")
        return jsonify({"error": "logs error"}), 500


@api_bp.get("/api/stats")
def stats_legacy():
    try:
        return jsonify(queries.api_stats(_db()))
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return jsonify({"error": "database error"}), 500


@api_bp.get("/api/csrf")
def csrf_token():
    token = _csrf_for_request()
    resp = make_response(jsonify({"csrf_token": token}))
    _ensure_csrf(resp)
    return resp
