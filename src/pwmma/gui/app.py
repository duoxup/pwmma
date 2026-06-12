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
    # Same diskcache as the background manager: a cross-process store so results
    # written by the background-callback worker are readable in the main process.
    app._pwmma_cache = cache
    app.layout = build_layout  # callable: re-read on each page load (saved defaults apply)
    cb.register_callbacks(app)
    cb.register_run_callback(app)
    cb.register_plot_callbacks(app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="pwmma-gui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="hot reload on code change (server errors go to the terminal)")
    parser.add_argument("--devtools", action="store_true",
                        help="with --debug: show Dash's floating dev-tools UI in the browser")
    args = parser.parse_args()
    app = create_app()
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            dev_tools_ui=args.debug and args.devtools)


if __name__ == "__main__":
    main()
