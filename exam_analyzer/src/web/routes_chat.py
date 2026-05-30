"""Chat assistant endpoints Blueprint.

Routes (7):
  GET  /api/chat/status           — chat availability check
  GET  /api/chat/history          — session chat history
  POST /api/chat                  — send question, get answer
  GET  /api/chat/exam-stats       — exam session statistics
  GET  /api/chat/student-state    — student memory state
  GET  /api/chat/student-confusions — detected confusion events
  GET  /api/chat/topic-questions  — practice questions per topic
"""

from flask import Blueprint

chat_bp = Blueprint("chat", __name__)

# Route implementations are currently in app.py.
# During P4 migration, each handler moves here.
# For now, the Blueprint is pre-registered in app_factory.py.
