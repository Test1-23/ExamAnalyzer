"""Quality evaluation endpoints Blueprint.

Routes (2):
  POST /api/evaluate        — start feedback agent evaluation
  GET  /api/evaluate/status — poll evaluation progress/result
"""

from flask import Blueprint

eval_bp = Blueprint("eval", __name__)

# Route implementations are currently in app.py.
# During P4 migration, each handler moves here.
# For now, the Blueprint is pre-registered in app_factory.py.
