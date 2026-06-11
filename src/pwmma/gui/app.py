# src/pwmma/gui/app.py
"""Dash app factory and `pwmma-gui` entry point."""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import webbrowser

import diskcache
from dash import Dash, DiskcacheManager

from . import callbacks as cb
from .layout import build_layout


def create_app() -> Dash:
    cache = diskcache.Cache(os.path.join(tempfile.gettempdir(), "pwmma-gui-cache"))
    manager = DiskcacheManager(cache)
    app = Dash(__name__, background_callback_manager=manager, title="pwmma")
    app.layout = build_layout()
    cb.register_callbacks(app)
    cb.register_run_callback(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="pwmma-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = create_app()
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
