import math
import os
import threading

from flask import Flask, render_template_string, request
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in Render Environment Variables.")

LLM_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite-preview"
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

# Smaller embedding size to reduce memory usage on Render Free.
EMBEDDING_DIM = int(
    os.getenv("GEMINI_EMBEDDING_DIM", "768")
)

TOP_K = int(
    os.getenv("RAG_TOP_K", "3")
)

CHUNK_SIZE = int(
    os.getenv("RAG_CHUNK_SIZE", "900")
)

app = Flask(__name__)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

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


# ============================================================
# CHUNKING
# ============================================================

def make_chunks(text, max_chars=900):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if not current:
            current = paragraph
            continue

        candidate = current + "\n\n" + paragraph

        if len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


CHUNKS = make_chunks(KT_GUIDE_CONTENT, CHUNK_SIZE)


# ============================================================
# LAZY GEMINI INITIALIZATION
# ============================================================

_embeddings = None
_llm = None
_document_vectors = None

_init_lock = threading.Lock()


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        with _init_lock:

            if _embeddings is None:

                _embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    output_dimensionality=EMBEDDING_DIM,
                )

    return _embeddings


def get_llm():

    global _llm

    if _llm is None:

        with _init_lock:

            if _llm is None:

                _llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0,
                )

    return _llm


# ============================================================
# VECTOR FUNCTIONS
# ============================================================

def normalize(vector):

    values = [
        float(x)
        for x in vector
    ]

    norm = math.sqrt(
        sum(x * x for x in values)
    )

    if norm == 0:
        return values

    return [
        x / norm
        for x in values
    ]


def cosine_similarity(a, b):

    return sum(
        x * y
        for x, y in zip(a, b)
    )


# ============================================================
# BUILD SMALL IN-MEMORY VECTOR INDEX
# ============================================================

def build_document_index():

    global _document_vectors

    if _document_vectors is not None:
        return

    with _init_lock:

        if _document_vectors is not None:
            return

        embeddings = get_embeddings()

        vectors = embeddings.embed_documents(
            CHUNKS
        )

        _document_vectors = [
            (
                chunk,
                normalize(vector)
            )
            for chunk, vector
            in zip(CHUNKS, vectors)
        ]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question):

    build_document_index()

    query_vector = get_embeddings().embed_query(
        question
    )

    query_vector = normalize(
        query_vector
    )

    scored = []

    for chunk, vector in _document_vectors:

        score = cosine_similarity(
            query_vector,
            vector
        )

        scored.append(
            (score, chunk)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk
        for _, chunk
        in scored[:TOP_K]
    ]


# ============================================================
# RAG GENERATION
# ============================================================

def answer_question(question):

    retrieved_chunks = retrieve(
        question
    )

    context = "\n\n".join(
        "[Source: InnovateCorp KT Guide]\n"
        + chunk
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are the InnovateCorp HR onboarding assistant.

Answer the user's question using ONLY the retrieved context below.

If the context does not contain enough information to answer, say:

"The information is not available in the InnovateCorp KT guide."

Do not invent facts.

Retrieved context:

{context}

User question:

{question}

Answer:
""".strip()

    response = get_llm().invoke(
        prompt
    )

    answer = response.content

    if isinstance(answer, list):

        answer = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    return str(answer), retrieved_chunks


# ============================================================
# HTML PAGE
# ============================================================

PAGE = """
<!doctype html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>
InnovateCorp RAG Assistant
</title>

<style>

body {

    margin: 0;

    background: #f4f6fa;

    color: #102a43;

    font-family: Arial, sans-serif;
}

.container {

    max-width: 1050px;

    margin: 36px auto;

    padding: 0 22px;
}

h1 {

    font-size: 40px;

    margin-bottom: 28px;
}

.subtitle {

    font-size: 21px;

    margin-bottom: 24px;
}

.card,
.section {

    background: white;

    border-radius: 18px;

    padding: 28px;

    box-shadow:
        0 8px 28px rgba(0,0,0,.08);
}

.section {

    margin-top: 24px;
}

label {

    display: block;

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 8px;
}

textarea {

    width: 100%;

    min-height: 125px;

    box-sizing: border-box;

    border: 1px solid #c7d2e0;

    border-radius: 12px;

    padding: 14px;

    font-size: 18px;

    resize: vertical;
}

button {

    margin-top: 20px;

    background: #2463eb;

    color: white;

    border: 0;

    border-radius: 11px;

    padding: 14px 23px;

    font-size: 17px;

    font-weight: 700;

    cursor: pointer;
}

.answer,
.source {

    white-space: pre-wrap;

    line-height: 1.6;
}

.source {

    border-top:
        1px solid #e2e8f0;

    margin-top: 14px;

    padding-top: 14px;
}

.error {

    color: #b42318;

    font-weight: 700;
}

small {

    color: #52606d;
}

</style>

</head>


<body>

<div class="container">

<h1>
InnovateCorp RAG Assistant
</h1>

<div class="subtitle">
Retrieval-Augmented Generation using
Gemini embeddings and Gemini.
</div>


<div class="card">

<form method="post">

<label for="question">
Ask a question about the KT guide
</label>

<textarea
    id="question"
    name="question"
    placeholder="Example: Who should I report to on Project Alpha?"
>{{ question }}</textarea>

<button type="submit">
Ask
</button>

</form>

</div>


{% if error %}

<div class="section">

<div class="error">
{{ error }}
</div>

</div>

{% endif %}


{% if answer %}

<div class="section">

<h2>
Answer
</h2>

<div class="answer">
{{ answer }}
</div>

</div>


<div class="section">

<h2>
Retrieved Context
</h2>

<small>
Chunks retrieved before generation.
</small>


{% for chunk in chunks %}

<div class="source">

<strong>
InnovateCorp KT Guide
</strong>

<br>

{{ chunk }}

</div>

{% endfor %}

</div>

{% endif %}

</div>

</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    question = ""

    answer = None

    chunks = []

    error = None

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if not question:

            error = "Please enter a question."

        else:

            try:

                answer, chunks = answer_question(
                    question
                )

            except Exception as exc:

                error = (
                    f"{type(exc).__name__}: {exc}"
                )

    return render_template_string(
        PAGE,

        question=question,

        answer=answer,

        chunks=chunks,

        error=error,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "llm_model": LLM_MODEL,

        "embedding_model": EMBEDDING_MODEL,

        "embedding_dim": EMBEDDING_DIM,

        "top_k": TOP_K,

        "index_ready":
            _document_vectors is not None,
    }


# ============================================================
# LOCAL START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
