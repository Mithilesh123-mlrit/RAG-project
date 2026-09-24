import math
import os
import threading
from typing import List, Tuple

from flask import Flask, render_template_string, request
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in the environment.")

LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)

# Free Render optimization:
# Gemini supports smaller embedding dimensions. 768 is much smaller than
# the default 3072-dimensional vector and is sufficient for this small demo.
EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))

app = Flask(__name__)

KT_GUIDE_CONTENT = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to
help new employees navigate their initial weeks and understand key aspects
of our operations. Our core values are Innovation, Collaboration, and
Customer Focus.

Team Structure: You will be joining the 'Project Alpha' team, reporting to
Sarah Chen, the Senior Project Manager. Your direct teammates include
David Lee (Lead Developer), Maria Rodriguez (UI/UX Designer), and Tom
Jackson (QA Engineer). Team meetings are held every Monday at 10 AM in
Conference Room 3, and daily stand-ups are at 9:30 AM via Google Meet.

Key Tools & Software: For project management, we use Jira for task tracking
and Confluence for documentation. Our primary communication tool is Slack
for instant messaging and Google Workspace for email and calendars.
Development work is primarily done using Python and JavaScript, with code
hosted on GitHub. Access to these tools will be granted within your first
three days.

Onboarding Process: Your first week will focus on setup and introductions.
You'll receive your laptop and login credentials on day one. HR will conduct
an orientation session on Tuesday covering company policies, benefits, and
payroll. You'll have one-on-one meetings with your team members throughout
the week. By the end of your second week, you should have access to all
necessary systems and have completed mandatory compliance training modules.

Important Resources: The company's internal knowledge base can be found at
internal.innovatecorp.com/kb. This includes FAQs, best practices, and
troubleshooting guides. For IT support, submit a ticket via
support.innovatecorp.com or call extension 5555. Health and wellness
benefits information is available on the HR portal.

Culture & Expectations: InnovateCorp encourages a proactive and collaborative
environment. We value open communication and continuous learning. Don't
hesitate to ask questions; your team is here to support your growth.
Performance reviews are conducted quarterly, and professional development
courses are available through the 'InnovateLearn' platform.
""".strip()


def make_documents(text: str) -> List[Document]:
    """Create small paragraph-based chunks without a heavy splitter package."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        extra = len(paragraph) + (2 if current else 0)

        if current and current_len + extra > CHUNK_SIZE:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + (2 if len(current) > 1 else 0)

    if current:
        chunks.append("\n\n".join(current))

    return [
        Document(
            page_content=chunk,
            metadata={"source": "InnovateCorp KT Guide"}
        )
        for chunk in chunks
    ]
