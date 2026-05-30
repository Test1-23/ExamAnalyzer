"""Analysis endpoints Blueprint.

Routes (7):
  GET  /api/config       — read/write API configuration
  GET  /api/files        — list input directory PDF files
  POST /api/analyze      — start analysis pipeline
  GET  /api/status       — poll running analysis status
  GET  /api/points       — read generated points.txt
  DELETE /api/input-files — clear input directory
  GET  /api/timeline     — execution timeline entries
"""

from flask import Blueprint

analysis_bp = Blueprint("analysis", __name__)

# Route implementations are currently in app.py.
# During P4 migration, each handler moves here.
# For now, the Blueprint is pre-registered in app_factory.py.
