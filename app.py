import os
import re
from functools import lru_cache

from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)

# Keep Flask lightweight
app.config["JSON_SORT_KEYS"] = False


# ============================================================
# Gemini configuration
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set"
    )

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ============================================================
# YOUR KNOWLEDGE BASE
# Replace this text with your actual knowledge base.
# Keep chunks reasonably small.
# ============================================================

KNOWLEDGE_BASE = """
Your knowledge base goes here.

Example:

Our company provides software development services.
We build web applications, mobile applications,
AI applications and RAG systems.

Our working hours are Monday to Friday,
9:00 AM to 6:00 PM.

For technical support, customers can contact
the support team through the official support email.

Replace this entire section with your actual content.
"""


# ============================================================
# Lightweight text chunking
# ============================================================

def create_chunks(text, max_words=180):
    """
    Split the knowledge base into small chunks.

    No embeddings.
    No vector database.
    No heavy ML model.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


CHUNKS = create_chunks(KNOWLEDGE_BASE)


# ============================================================
# Very lightweight tokenizer
# ============================================================

def tokenize(text):
    """
    Convert text into simple lowercase keywords.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]{2,}\b",
            text.lower()
        )
    )


# Pre-tokenize chunks once.
# This happens only when the application starts.
CHUNK_TOKENS = [
    tokenize(chunk)
    for chunk in CHUNKS
]


# ============================================================
# Lightweight RAG retrieval
# ============================================================

def retrieve_context(question, top_k=3):
    """
    Lightweight retrieval using keyword overlap.

    This uses almost no RAM compared with an embedding model
    and vector database.
    """

    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    scored_chunks = []

    for index, chunk_tokens in enumerate(CHUNK_TOKENS):

        overlap = question_tokens.intersection(chunk_tokens)

        if not overlap:
            continue

        # Simple relevance score
        score = len(overlap) / max(len(question_tokens), 1)

        scored_chunks.append(
            (score, index)
        )

    # Highest score first
    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    for score, index in scored_chunks[:top_k]:

        # Ignore extremely weak matches
        if score >= 0.05:
            selected.append(CHUNKS[index])

    return selected


# ============================================================
# Gemini model
# ============================================================

@lru_cache(maxsize=1)
def get_llm():

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
        max_output_tokens=512,
    )


# ============================================================
# RAG answer
# ============================================================

def generate_answer(question):
    context = retrieve_context(question)

    prompt = f"""
Answer the user's question using the knowledge base below.

Knowledge base:
{context}

User question:
{question}

Give a clear and concise answer.
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    # Extract plain text from Gemini/LangChain response
    if hasattr(response, "text"):
        return response.text

    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        return str(content)

    return str(response)


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "ok",
        "chunks": len(CHUNKS),
        "model": GEMINI_MODEL
    })


# ============================================================
# Main RAG API
# ============================================================

@app.post("/ask")
def ask():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "error": "Question is required"
            }), 400

        # Prevent unnecessarily huge requests
        if len(question) > 2000:

            return jsonify({
                "error": "Question is too long"
            }), 400

        answer = generate_answer(question)

        return jsonify({
            "answer": answer
        })

    except Exception as exc:

        app.logger.exception(
            "RAG request failed"
        )

        return jsonify({
            "error": "Internal server error",
            "details": str(exc)
        }), 500


# ============================================================
# Optional browser UI
# ============================================================

@app.get("/")
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Assistant</title>

        <meta name="viewport"
              content="width=device-width, initial-scale=1">

        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
            }

            textarea {
                width: 100%;
                height: 100px;
                padding: 12px;
                font-size: 16px;
                box-sizing: border-box;
            }

            button {
                margin-top: 10px;
                padding: 12px 20px;
                cursor: pointer;
            }

            #answer {
                margin-top: 25px;
                white-space: pre-wrap;
            }

            #loading {
                display: none;
            }
        </style>
    </head>

    <body>

        <h1>RAG Assistant</h1>

        <textarea
            id="question"
            placeholder="Ask a question..."
        ></textarea>

        <br>

        <button onclick="askQuestion()">
            Ask
        </button>

        <span id="loading">
            Loading...
        </span>

        <div id="answer"></div>

        <script>

        async function askQuestion() {

            const question =
                document.getElementById("question").value.trim();

            const answer =
                document.getElementById("answer");

            const loading =
                document.getElementById("loading");

            if (!question) {
                answer.innerText =
                    "Please enter a question.";
                return;
            }

            answer.innerText = "";
            loading.style.display = "inline";

            try {

                const response = await fetch(
                    "/ask",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question: question
                        })
                    }
                );

                const data =
                    await response.json();

                if (!response.ok) {

                    answer.innerText =
                        data.error ||
                        "Request failed.";

                } else {

                    answer.innerText =
                        data.answer || "";
                }

            } catch (error) {

                answer.innerText =
                    "Unable to connect to the server.";

            } finally {

                loading.style.display = "none";
            }
        }

        </script>

    </body>
    </html>
    """


# ============================================================
# Local development only
# Render uses Gunicorn.
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get("PORT", 5000)
        ),
        debug=False
    )
