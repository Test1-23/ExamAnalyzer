"""Flask application factory — creates and configures the Flask app.

Usage::

    from src.web.app_factory import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=5000)

Blueprints are registered here so app.py becomes a thin entry point.
"""

import os
import time

from flask import Flask

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)  # src/web/ → src/ → exam_analyzer/


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_DIR, "templates"),
        static_folder=os.path.join(PROJECT_DIR, "static")
        if os.path.isdir(os.path.join(PROJECT_DIR, "static"))
        else None,
    )

    # ---- Request logging middleware ----
    @app.before_request
    def _log_request():
        from flask import request
        request._start_time = time.time()

    @app.after_request
    def _log_response(response):
        from flask import request
        from src.logger import get_logger
        _log = get_logger()
        elapsed = int((time.time() - getattr(request, '_start_time', time.time())) * 1000)
        _log.debug(f"HTTP {request.method} {request.path} → {response.status_code} ({elapsed}ms)")
        return response

    # ---- Register Blueprints (routes currently in app.py) ----
    from .routes_analysis import analysis_bp
    from .routes_chat import chat_bp
    from .routes_eval import eval_bp
    from .routes_knowledge import knowledge_bp

    app.register_blueprint(analysis_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(eval_bp)
    app.register_blueprint(knowledge_bp)

    return app
