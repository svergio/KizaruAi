"""HTML views for Admin v2."""
from __future__ import annotations

import logging
import sqlite3

from flask import Blueprint, g, redirect, render_template, request

from app.web import queries
from app.web.contracts import KIND, STATUS, WORKER_KIND

log = logging.getLogger("kizaru.web")
views_bp = Blueprint("views", __name__)


@views_bp.after_request
def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    if request.method == "HEAD":
        resp.set_data(b"")
    return resp


def _ctx(**extra):
    # UI JSON must not embed hex colors (tokens live in admin.css).
    status_ui = {k: {"label": v["label"], "css": v["css"]} for k, v in STATUS.items()}
    base = {
        "status_map": status_ui,
        "kind_map": KIND,
        "worker_kind_map": WORKER_KIND,
        "refresh_sec": 3,
        "db_ok": True,
        "guilds": [],
    }
    base.update(extra)
    return base


@views_bp.get("/")
@views_bp.get("/index.html")
def home_page():
    db_path = g.db_path
    db_ok = db_path.is_file()
    try:
        if db_ok:
            # touch read path so sqlite errors surface as HTML DB error
            queries.api_home(db_path)
            db_ok = True
    except sqlite3.Error:
        log.exception("admin sqlite error")
        return (
            '<!DOCTYPE html><html lang="ru"><body><h1>Ошибка</h1>'
            "<p>Ошибка базы данных</p></body></html>",
            503,
            {"Content-Type": "text/html; charset=utf-8"},
        )
    return render_template(
        "home.html",
        **_ctx(active_tab="home", title="Главная", db_ok=db_ok),
    )


@views_bp.get("/jobs")
def jobs_page():
    status = (request.args.get("status") or "").strip()
    kind = (request.args.get("kind") or "").strip()
    user = (request.args.get("user") or "").strip()
    try:
        page = max(1, int(request.args.get("page") or 1))
    except ValueError:
        page = 1
    return render_template(
        "jobs.html",
        **_ctx(
            active_tab="jobs",
            title="Задачи",
            filter_status=status,
            filter_kind=kind,
            filter_user=user,
            filter_page=page,
        ),
    )


@views_bp.get("/jobs/<int:job_id>")
def job_page(job_id: int):
    return render_template(
        "job_detail.html", **_ctx(active_tab="jobs", title=f"Задача #{job_id}", job_id=job_id)
    )


@views_bp.get("/users")
def users_page():
    return render_template("users.html", **_ctx(active_tab="users", title="Пользователи"))


@views_bp.get("/users/<path:user_id>")
def user_page(user_id: str):
    return render_template(
        "user_detail.html",
        **_ctx(active_tab="users", title=f"Пользователь {user_id}", user_id=user_id),
    )


@views_bp.get("/workers")
def workers_page():
    return render_template("workers.html", **_ctx(active_tab="workers", title="Воркеры"))


@views_bp.get("/models")
def models_redirect():
    return redirect("/workers", code=302)


@views_bp.get("/logs")
def logs_page():
    return render_template("logs.html", **_ctx(active_tab="logs", title="Логи"))


@views_bp.get("/queue")
def queue_redirect():
    return redirect("/jobs", code=302)


@views_bp.get("/services")
def services_page():
    db_path = g.db_path
    db_ok = db_path.is_file()
    return render_template(
        "services.html",
        **_ctx(active_tab="services", title="Сервисы", db_ok=db_ok),
    )


@views_bp.get("/system")
def system_page():
    db_path = g.db_path
    db_ok = db_path.is_file()
    return render_template(
        "system.html",
        **_ctx(active_tab="system", title="Система", db_ok=db_ok),
    )
