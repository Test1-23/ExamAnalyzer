"""Web layer package — Flask Blueprint-based restructuring (P4 complete).

Public API:
    from src.web import create_app, state
    from src.web.routes_analysis import analysis_bp
    from src.web.routes_chat import chat_bp
    from src.web.routes_eval import eval_bp
    from src.web.routes_knowledge import knowledge_bp
"""

from .app_factory import create_app

__all__ = ["create_app"]
