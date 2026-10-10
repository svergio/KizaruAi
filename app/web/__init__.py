"""Flask application factory for Admin v2."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, g, jsonify

from app.web.queries import DEFAULT_DB, ROOT


def create_app(db_path: Path | str | None = None) -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static",
        template_folder="templates",
    )
    app.config["DEBUG"] = False
    app.config["TESTING"] = False
    path = Path(db_path) if db_path is not None else DEFAULT_DB
    if not path.is_absolute():
        path = ROOT / path
    app.config["DB_PATH"] = path

    from app.web.api import api_bp
    from app.web.views import views_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    @app.before_request
    def _bind_db():
        g.db_path = app.config["DB_PATH"]

    @app.errorhandler(404)
    def _not_found(err):
        from flask import request

        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return (
            '<!DOCTYPE html><html lang="ru"><body><h1>404</h1>'
            "<p>Страница не найдена</p></body></html>",
            404,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.errorhandler(500)
    def _server_error(err):
        from flask import request
        import logging
        import sqlite3

        logging.getLogger("kizaru.web").exception("admin error")
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal error"}), 500
        orig = getattr(err, "original_exception", None) or err
        if isinstance(orig, sqlite3.Error) or isinstance(
            getattr(err, "description", None), sqlite3.Error
        ):
            return (
                '<!DOCTYPE html><html lang="ru"><body><h1>Ошибка</h1>'
                "<p>Ошибка базы данных</p></body></html>",
                503,
                {"Content-Type": "text/html; charset=utf-8"},
            )
        return (
            '<!DOCTYPE html><html lang="ru"><body><h1>Ошибка</h1>'
            "<p>Ошибка сервера</p></body></html>",
            500,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.errorhandler(503)
    def _unavailable(err):
        from flask import request

        if request.path.startswith("/api/"):
            return jsonify({"error": "database error"}), 503
        return (
            '<!DOCTYPE html><html lang="ru"><body><h1>Ошибка</h1>'
            "<p>Ошибка базы данных</p></body></html>",
            503,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    return app
