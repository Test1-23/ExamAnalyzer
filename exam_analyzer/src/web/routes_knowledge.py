"""Knowledge base & practice endpoints Blueprint.

Routes (5):
  POST /api/practice/generate  — generate practice questions
  POST /api/practice/grade     — grade practice answer
  GET  /api/knowledge-graph    — knowledge graph data
  GET  /api/command-verbs      — command verb analysis
  GET  /api/topic-difficulty   — topic difficulty assessment
"""

from flask import Blueprint

knowledge_bp = Blueprint("knowledge", __name__)

# Route implementations are currently in app.py.
# During P4 migration, each handler moves here.
# For now, the Blueprint is pre-registered in app_factory.py.
