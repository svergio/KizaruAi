"""python -m app.web"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.web import create_app
from app.web.queries import DEFAULT_DB, DEFAULT_HOST, DEFAULT_PORT, ROOT
from core.logging_setup import configure_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Локальная панель KizaruAi Admin v2")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Адрес (только локальный)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Порт HTTP")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Путь к SQLite bot.db")
    parser.add_argument(
        "--server",
        choices=("waitress", "flask"),
        default="waitress",
        help="WSGI-сервер",
    )
    return parser.parse_args(argv)


def main(argv=None):
    configure_logging("web")
    args = parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Разрешён только локальный bind (127.0.0.1).", file=sys.stderr)
        sys.exit(1)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    app = create_app(db_path)
    print(f"admin panel: http://{args.host}:{args.port}/ db={db_path}")
    if args.server == "flask":
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
        return
    try:
        from waitress import serve
    except ImportError:
        msg = "waitress не установлен; fallback на flask threaded\n"
        sys.stderr.buffer.write(msg.encode("utf-8"))
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
        return
    serve(app, host=args.host, port=args.port, threads=8)


if __name__ == "__main__":
    main()
