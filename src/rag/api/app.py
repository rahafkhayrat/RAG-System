from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import UploadFile
from werkzeug.utils import secure_filename

from rag.config import Settings
from rag.embeddings.hashing import HashingEmbedder
from rag.generation.llm import OpenAICompatibleLLM
from rag.pipeline import RagEngine
from rag.store.memory import InMemoryVectorStore
from .routes.health import health_router
from .routes.questions import questions_router


ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Course RAG Assistant</title>

<!-- Markdown is sanitized before insertion; KaTeX renders mathematical notation. -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>

<style>
* { box-sizing: border-box; }

body {
  margin: 0;
  background: #f4f6fa;
  color: #202938;
  font: 16px/1.6 system-ui, sans-serif;
}

main {
  max-width: 900px;
  margin: 40px auto;
  padding: 20px;
}

header { margin-bottom: 28px; }
h1 { margin-bottom: 5px; }
.muted { color: #667085; }

.card {
  background: white;
  border: 1px solid #e1e6ef;
  border-radius: 14px;
  padding: 22px;
  margin: 18px 0;
}

label {
  display: block;
  font-weight: 600;
  margin: 12px 0 5px;
}

input, textarea, button {
  width: 100%;
  font: inherit;
  padding: 11px 12px;
  border: 1px solid #ccd3df;
  border-radius: 8px;
}

textarea {
  min-height: 100px;
  resize: vertical;
}

button {
  margin-top: 14px;
  background: #273b70;
  color: white;
  border: 0;
  cursor: pointer;
  font-weight: 600;
}

button:disabled {
  opacity: .6;
  cursor: wait;
}

.status {
  margin-top: 12px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.answer {
  overflow-wrap: anywhere;
  line-height: 1.75;
}
.answer p { margin: 0 0 1em; }
.answer h1, .answer h2, .answer h3, .answer h4 { margin: 1em 0 .5em; line-height: 1.35; }
.answer ul, .answer ol { padding-left: 1.5rem; margin: .5em 0 1em; }
.answer pre {
  overflow-x: auto;
  padding: 12px;
  border-radius: 8px;
  background: #f4f6fa;
}
.answer code {
  overflow-wrap: anywhere;
  background: #f4f6fa;
  border-radius: 4px;
  padding: 2px 4px;
}
.answer pre code { padding: 0; }
.answer blockquote {
  border-left: 3px solid #ccd3df;
  margin-left: 0;
  padding-left: 1em;
  color: #475467;
}
.answer .katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 6px 0;
}
.answer table { border-collapse: collapse; display: block; overflow-x: auto; }
.answer th, .answer td { border: 1px solid #ccd3df; padding: 6px 10px; }

.source {
  border-top: 1px solid #e7eaf0;
  padding: 12px 0;
  overflow-wrap: anywhere;
}

.score {
  color: #667085;
  font-size: 13px;
}
</style>
</head>

<body>
<main>

<header>
  <h1>Course RAG Assistant</h1>
  <div class="muted">
    Upload course materials, then ask questions grounded in them.
  </div>
</header>

<section class="card">
  <h2>1. Upload course materials</h2>

  <form id="uploadForm">
    <label for="course">Course ID</label>
    <input
      id="course"
      name="course_id"
      value="course-1"
      required
      pattern="[A-Za-z0-9_-]+"
      maxlength="80"
    >

    <label for="files">PDF or DOCX files (20 MB total maximum)</label>
    <input
      id="files"
      name="files"
      type="file"
      accept=".pdf,.docx"
      multiple
      required
    >

    <button id="uploadButton" type="submit">
      Upload and index files
    </button>
  </form>

  <div id="uploadStatus" class="status" role="status"></div>
</section>

<section class="card">
  <h2>2. Ask a question</h2>

  <form id="questionForm">
    <label for="questionCourse">Course ID</label>
    <input
      id="questionCourse"
      value="course-1"
      required
      pattern="[A-Za-z0-9_-]+"
      maxlength="80"
    >

    <label for="question">Question</label>
    <textarea
      id="question"
      required
      maxlength="3000"
      placeholder="Ask something covered in your course materials..."
    ></textarea>

    <button id="askButton" type="submit">Ask question</button>
  </form>

  <div id="answerStatus" class="status" role="status"></div>

  <div id="answerArea" hidden>
    <h3>Answer</h3>
    <div id="answer" class="answer"></div>

    <h3>Sources</h3>
    <div id="sources"></div>
  </div>
</section>

</main>

<script>
const uploadForm = document.getElementById("uploadForm");
const questionForm = document.getElementById("questionForm");

/* Upload one or multiple course documents */

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const button = document.getElementById("uploadButton");
    const status = document.getElementById("uploadStatus");
    const courseId = document.getElementById("course").value.trim();
    const files = Array.from(
        document.getElementById("files").files
    );

    if (!courseId) {
        status.textContent = "Please provide a Course ID.";
        return;
    }

    if (files.length === 0) {
        status.textContent = "Please select at least one file.";
        return;
    }

    const allowedExtensions = [".pdf", ".docx"];

    for (const file of files) {
        const filename = file.name.toLowerCase();

        if (!allowedExtensions.some(
            extension => filename.endsWith(extension)
        )) {
            status.textContent =
                `Unsupported file type: ${file.name}. Use PDF or DOCX.`;
            return;
        }
    }

    const totalBytes = files.reduce(
        (total, file) => total + file.size,
        0
    );

    if (totalBytes > 20 * 1024 * 1024) {
        status.textContent =
            "The selected files exceed the 20 MB total upload limit.";
        return;
    }

    const formData = new FormData();
    formData.append("course_id", courseId);

    for (const file of files) {
        formData.append("files", file);
    }

    button.disabled = true;

    status.textContent =
        `Uploading ${files.length} file(s) and indexing their content...`;

    try {
        const response = await fetch("/api/documents", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok && !result.files) {
            throw new Error(result.error || "Upload failed.");
        }

        document.getElementById("questionCourse").value = courseId;

        const lines = [
            `Course ID: ${result.course_id}`,
            `Total files: ${result.total_files}`,
            `Successfully indexed: ${result.successful_files}`,
            `Failed: ${result.failed_files}`,
            ""
        ];

        for (const file of result.files || []) {
            if (file.error) {
                lines.push(`FAILED: ${file.filename}`);
                lines.push(`Reason: ${file.error}`);
            } else {
                lines.push(`SUCCESS: ${file.filename}`);
                lines.push(`Chunks indexed: ${file.chunks}`);
            }

            lines.push("");
        }

        status.textContent = lines.join("\n");

    } catch (error) {
        status.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
    }
});


/* Ask questions across the indexed documents in a course */

questionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const button = document.getElementById("askButton");
    const status = document.getElementById("answerStatus");
    const area = document.getElementById("answerArea");
    const sources = document.getElementById("sources");

    const courseId = document
        .getElementById("questionCourse")
        .value.trim();

    const question = document
        .getElementById("question")
        .value.trim();

    if (!courseId || !question) {
        status.textContent = "Course ID and question are required.";
        return;
    }

    button.disabled = true;
    area.hidden = true;
    sources.replaceChildren();

    status.textContent = "Searching course materials...";

    try {
        const response = await fetch("/api/questions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                course_id: courseId,
                question: question
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Question failed.");
        }

        const answerElement = document.getElementById("answer");
        const rawAnswer = result.answer || result.message || "No answer generated.";

        // Some models escape Markdown markers unnecessarily (\*\*).
        // Normalize those markers without altering LaTeX commands.
        let markdownAnswer = String(rawAnswer).replace(/\\\*\\\*/g, "**");

        if (
            window.marked &&
            window.DOMPurify &&
            window.renderMathInElement &&
            window.katex
        ) {
            // Protect math expressions BEFORE Markdown parsing. Markdown parsers
            // can consume backslashes in \[...\] and \( ... \), breaking KaTeX.
            const mathExpressions = [];
            const protectMath = (match) => {
                const token = `MATHPLACEHOLDER${mathExpressions.length}TOKEN`;
                mathExpressions.push({ token, source: match });
                return token;
            };

            markdownAnswer = markdownAnswer
                .replace(/\\\[([\s\S]*?)\\\]/g, protectMath)
                .replace(/\\\(([\s\S]*?)\\\)/g, protectMath)
                .replace(/\$\$([\s\S]*?)\$\$/g, protectMath)
                .replace(/\$([^\n$]+?)\$/g, protectMath);

            const renderedMarkdown = window.marked.parse(markdownAnswer, {
                gfm: true,
                breaks: true
            });

            // Sanitize untrusted LLM output before adding it to the DOM.
            answerElement.innerHTML = window.DOMPurify.sanitize(renderedMarkdown);

            // Restore protected math as TEXT nodes, not HTML, then let KaTeX render it.
            const walker = document.createTreeWalker(
                answerElement,
                NodeFilter.SHOW_TEXT
            );
            const textNodes = [];
            while (walker.nextNode()) textNodes.push(walker.currentNode);

            for (const node of textNodes) {
                let value = node.nodeValue;
                let changed = false;
                for (const entry of mathExpressions) {
                    if (value.includes(entry.token)) {
                        value = value.replace(entry.token, entry.source);
                        changed = true;
                    }
                }
                if (changed) node.nodeValue = value;
            }

            window.renderMathInElement(answerElement, {
                delimiters: [
                    { left: "$$", right: "$$", display: true },
                    { left: "\\[", right: "\\]", display: true },
                    { left: "\\(", right: "\\)", display: false }
                ],
                throwOnError: false,
                strict: "ignore",
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
            });
        } else {
            // Graceful fallback if CDN libraries are unavailable.
            answerElement.textContent = rawAnswer;
        }

        status.textContent = result.abstained
            ? `No generated answer (${result.reason}).`
            : "Answer generated.";

        for (const source of result.sources || []) {
            const item = document.createElement("div");
            item.className = "source";

            const heading = document.createElement("strong");
            heading.textContent =
                source.filename || source.source_uri || "Source";

            item.appendChild(heading);

            const location = document.createElement("div");
            location.textContent =
                `Page: ${source.page ?? "unknown"} · ` +
                `Section: ${source.section || "—"}`;

            item.appendChild(location);

            const score = document.createElement("div");
            score.className = "score";

            const similarity = Number(source.score);

            score.textContent = Number.isFinite(similarity)
                ? `Similarity score: ${similarity.toFixed(3)}`
                : "Similarity score: unavailable";

            item.appendChild(score);

            const excerpt = document.createElement("p");
            excerpt.textContent = source.text || "";

            item.appendChild(excerpt);
            sources.appendChild(item);
        }

        area.hidden = false;

    } catch (error) {
        status.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
    }
});
</script>

</body>
</html>
"""


def build_engine(settings: Settings) -> RagEngine:
    """Create the local engine without downloading an embedding model by default."""

    model_name = os.getenv("RAG_EMBED_MODEL", "hashing").strip()

    if model_name.lower() == "hashing":
        embedder = HashingEmbedder()
    else:
        from rag.embeddings.bge_m3 import BgeM3Embedder

        embedder = BgeM3Embedder(model_name=model_name)

    llm = None

    if settings.has_llm():
        llm = OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

    return RagEngine(
        store=InMemoryVectorStore(),
        embedder=embedder,
        llm=llm,
        top_k=settings.top_k,
    )


def create_app(engine: RagEngine | None = None) -> FastAPI:
    app = FastAPI()

    settings = Settings.from_env()
    rag_engine = engine or build_engine(settings)

    tenant_id = os.getenv("RAG_TENANT_ID", "local-user")

    app.state.rag_tenant_id = tenant_id
    app.state.rag_engine = rag_engine
    app.state.llm_configured = (
        settings.has_llm()
        if engine is None
        else getattr(rag_engine, "llm", None) is not None
    )

    app.state.embedding_model = os.getenv(
        "RAG_EMBED_MODEL",
        "hashing",
    )

    app.include_router(health_router)
    app.include_router(questions_router)

    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return HTMLResponse(HTML)

    @app.middleware("http")
    async def enforce_upload_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                exceeds_limit = int(content_length) > MAX_UPLOAD_BYTES
            except ValueError:
                exceeds_limit = False

            if exceeds_limit:
                return JSONResponse(
                    {
                        "error": (
                            "The total upload request exceeds the 20 MB limit."
                        )
                    },
                    status_code=413,
                )

        return await call_next(request)

    @app.post("/api/documents")
    async def upload_document(request: Request):
        form = await request.form()
        course_id_value = form.get("course_id", "")
        course_id = (
            course_id_value.strip()
            if isinstance(course_id_value, str)
            else ""
        )

        if not course_id or len(course_id) > 80:
            return JSONResponse(
                {"error": "Provide a valid course_id."},
                status_code=400,
            )

        if not all(ch.isalnum() or ch in "_-" for ch in course_id):
            return JSONResponse(
                {
                    "error": (
                        "course_id may contain only letters, "
                        "numbers, _ and -."
                    )
                },
                status_code=400,
            )

        uploaded_files = [
            item for item in form.getlist("files")
            if isinstance(item, UploadFile)
        ]

        # Preserve compatibility with the original single-file API.
        if not uploaded_files:
            old_file = form.get("file")

            if isinstance(old_file, UploadFile):
                uploaded_files = [old_file]

        if not uploaded_files or not any(
            uploaded.filename for uploaded in uploaded_files
        ):
            return JSONResponse(
                {"error": "Choose at least one PDF or DOCX file."},
                status_code=400,
            )

        # Preserve the previous API status code for a single
        # unsupported extension (e.g. notes.txt).
        if len(uploaded_files) == 1:
            original_name = uploaded_files[0].filename or ""
            safe_name = secure_filename(original_name)
            suffix = Path(safe_name).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                return JSONResponse(
                    {"error": "Only PDF and DOCX files are supported."},
                    status_code=415,
                )

        results = []

        for uploaded in uploaded_files:
            original_name = (uploaded.filename or "").strip()

            if not original_name:
                results.append({
                    "filename": "Unknown file",
                    "error": "No file was selected.",
                })
                continue

            safe_name = secure_filename(original_name)

            if not safe_name:
                results.append({
                    "filename": original_name,
                    "error": "Invalid filename.",
                })
                continue

            suffix = Path(safe_name).suffix.lower()

            if suffix not in ALLOWED_EXTENSIONS:
                results.append({
                    "filename": safe_name,
                    "error": "Only PDF and DOCX files are supported.",
                })
                continue

            document_id = uuid.uuid4().hex
            stored_path = upload_dir / f"{document_id}{suffix}"

            try:
                with stored_path.open("wb") as destination:
                    shutil.copyfileobj(uploaded.file, destination)

                chunks = rag_engine.ingest(
                    stored_path,
                    tenant_id=tenant_id,
                    course_id=course_id,
                    document_id=document_id,
                    source_uri=safe_name,
                )

                logger.info(
                    "Indexed document chunks=%s tenant=%s course=%s",
                    chunks,
                    tenant_id,
                    course_id,
                )

                if chunks == 0:
                    stored_path.unlink(missing_ok=True)

                    results.append({
                        "filename": safe_name,
                        "error": (
                            "No indexable text was found "
                            "in this document."
                        ),
                    })
                    continue

                results.append({
                    "document_id": document_id,
                    "filename": safe_name,
                    "course_id": course_id,
                    "chunks": chunks,
                })

            except (ValueError, OSError) as exc:
                stored_path.unlink(missing_ok=True)

                results.append({
                    "filename": safe_name,
                    "error": str(exc),
                })

            except Exception:
                stored_path.unlink(missing_ok=True)

                logger.exception(
                    "Document ingestion failed for %s",
                    safe_name,
                )

                results.append({
                    "filename": safe_name,
                    "error": (
                        "Document ingestion failed. "
                        "Check the server log."
                    ),
                })

        successful_results = [
            item for item in results if "error" not in item
        ]

        failed_files = len(results) - len(successful_results)

        response_data = {
            "course_id": course_id,
            "total_files": len(results),
            "successful_files": len(successful_results),
            "failed_files": failed_files,
            "files": results,
        }

        # Backward compatibility with the original API response.
        if len(results) == 1 and successful_results:
            response_data.update({
                "document_id": successful_results[0]["document_id"],
                "filename": successful_results[0]["filename"],
                "chunks": successful_results[0]["chunks"],
            })

        status_code = 201 if successful_results else 422

        return JSONResponse(response_data, status_code=status_code)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
    )
