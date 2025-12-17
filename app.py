import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from dotenv import load_dotenv

# ---------------- LangChain (STABLE) ----------------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ---------------- OpenAI (A4F Native Client) ----------------
from openai import OpenAI

# ---------------- ENV ----------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# ---------------- OAUTH CONFIG ----------------
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
]
REDIRECT_URI = "http://localhost:5000/callback"

# ---------------- GLOBAL STATE ----------------
vectorstore = None
current_docs_content = []

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def build_vectorstore(docs_contents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = []
    for content in docs_contents:
        chunks.extend(splitter.split_text(content))
    return FAISS.from_texts(chunks, embedding=embeddings)

# ---------------- LLM (A4F Native) ----------------
def build_llm_client():
    return OpenAI(
        api_key=os.getenv("A4F_API_KEY"),
        base_url=os.getenv("A4F_BASE_URL")
    )

def llm_invoke(prompt: str) -> str:
    client = build_llm_client()
    completion = client.chat.completions.create(
        model="provider-5/gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return completion.choices[0].message.content

# ======================= ROUTES =======================

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(prompt="consent")
    session["state"] = state
    return redirect(auth_url)

@app.route("/callback")
def callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=session.get("state"),
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials

    session["credentials"] = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "id_token": credentials.id_token
    }

    user_info = id_token.verify_oauth2_token(
        credentials.id_token,
        grequests.Request(),
        audience=credentials.client_id
    )

    return render_template(
        "success.html",
        user_email=user_info.get("email", "User")
    )

@app.route("/dashboard")
def dashboard():
    if "credentials" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.clear()
    global vectorstore, current_docs_content
    vectorstore = None
    current_docs_content = []
    return redirect(url_for("index"))

# ---------------- GOOGLE DOCS ----------------

@app.route("/docs_json")
def docs_json():
    creds_info = session.get("credentials")
    if not creds_info:
        return jsonify({"files": []})

    credentials = google.oauth2.credentials.Credentials(**creds_info)
    drive_service = build("drive", "v3", credentials=credentials)

    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        pageSize=50,
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc"
    ).execute()

    return jsonify({"files": results.get("files", [])})

@app.route("/load_doc", methods=["POST"])
def load_doc():
    global vectorstore, current_docs_content
    data = request.json
    doc_ids = data.get("file_ids", [])

    if not doc_ids:
        return jsonify({"status": "No document selected"}), 400

    creds_info = session.get("credentials")
    if not creds_info:
        return jsonify({"status": "Not authenticated"}), 401

    credentials = google.oauth2.credentials.Credentials(**creds_info)
    docs_service = build("docs", "v1", credentials=credentials)

    current_docs_content = []
    titles = []

    for doc_id in doc_ids:
        doc = docs_service.documents().get(documentId=doc_id).execute()
        text = []
        for el in doc.get("body", {}).get("content", []):
            if "paragraph" in el:
                for run in el["paragraph"]["elements"]:
                    if "textRun" in run:
                        text.append(run["textRun"]["content"])

        content = "\n".join(text).strip()
        if content:
            current_docs_content.append(content)
            titles.append(doc.get("title", "Untitled"))

    if not current_docs_content:
        return jsonify({"status": "Documents empty"}), 400

    vectorstore = build_vectorstore(current_docs_content)
    return jsonify({
        "status": f"Loaded {len(titles)} document(s): {', '.join(titles)}"
    })

# ---------------- RAG CHATBOT ----------------

@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    query = request.json.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please enter a question"}), 400

    # Fallback: no documents loaded
    if not vectorstore:
        answer = llm_invoke(query)
        return jsonify({
            "answer": f"📚 General Answer:\n\n{answer}"
        })

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Use ONLY the context to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    )

    llm_runnable = RunnableLambda(
        lambda x: llm_invoke(x.to_string())
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_runnable
    )

    answer = rag_chain.invoke(query)

    return jsonify({
        "answer": f"📄 Answer based on your documents:\n\n{answer}"
    })

# ---------------- HEALTH ----------------

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None
    })

# ---------------- MAIN ----------------

if __name__ == "__main__":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    required = [
        "SECRET_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "A4F_API_KEY",
        "A4F_BASE_URL"
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    if not os.path.exists(CLIENT_SECRETS_FILE):
        raise RuntimeError("credentials.json not found")

    print("🚀 RAG Chatbot running at http://localhost:5000")
    app.run(debug=True)
