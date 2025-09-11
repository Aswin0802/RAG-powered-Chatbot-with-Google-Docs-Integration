import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.oauth2.credentials
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# OAuth Config
CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
]
REDIRECT_URI = "http://localhost:5000/callback"

# Global storage
vectorstore = None
current_docs_content = []

# Hugging Face embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def build_vectorstore(docs_contents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for content in docs_contents:
        chunks = text_splitter.split_text(content)
        all_chunks.extend(chunks)
    return FAISS.from_texts(all_chunks, embedding=embeddings)

# A4F API config
A4F_API_KEY = "ddc-a4f-0a2df298453d426faec81547e29a02ca"
A4F_BASE_URL = "https://api.a4f.co/v1"

def build_qa_chain(vstore):
    if not vstore:
        return None, None
    llm = ChatOpenAI(
        model="provider-3/gpt-4o-mini",
        temperature=0.5,
        openai_api_key=A4F_API_KEY,
        openai_api_base=A4F_BASE_URL
    )
    retriever = vstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return llm, retriever

#Routes

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
    state = session.get("state")
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
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

    # Extract user info safely from ID token
    user_info = id_token.verify_oauth2_token(
        credentials.id_token,
        grequests.Request(),
        audience=credentials.client_id
    )
    user_email = user_info.get("email", "User")

    return render_template("success.html", user_email=user_email)

@app.route("/dashboard")
def dashboard():
    creds_info = session.get("credentials")
    if not creds_info:
        return redirect(url_for("index"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.clear()
    global vectorstore, current_docs_content
    vectorstore = None
    current_docs_content = []
    return redirect(url_for("index"))

# Google Docs APIs 
@app.route("/docs_json")
def docs_json():
    creds_info = session.get("credentials")
    if not creds_info:
        return jsonify({"files": []})

    try:
        credentials = google.oauth2.credentials.Credentials(**creds_info)
        drive_service = build("drive", "v3", credentials=credentials)
        results = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.document'",
            pageSize=50,
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc"
        ).execute()
        files = results.get("files", [])
        return jsonify({"files": files})
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return jsonify({"files": [], "error": str(e)})
    

#Loading Docs
@app.route("/load_doc", methods=["POST"])
def load_doc():
    global vectorstore, current_docs_content
    data = request.json
    doc_ids = data.get("file_ids", [])
    if not doc_ids:
        return jsonify({"status": "No document selected."}), 400

    creds_info = session.get("credentials")
    if not creds_info:
        return jsonify({"status": "User not authenticated."}), 401

    try:
        credentials = google.oauth2.credentials.Credentials(**creds_info)
        docs_service = build("docs", "v1", credentials=credentials)

        current_docs_content = []
        loaded_titles = []

        for doc_id in doc_ids:
            try:
                doc = docs_service.documents().get(documentId=doc_id).execute()
                content = []
                for element in doc.get("body", {}).get("content", []):
                    if "paragraph" in element:
                        for text_run in element["paragraph"]["elements"]:
                            if "textRun" in text_run:
                                content.append(text_run["textRun"]["content"])
                doc_content = "\n".join(content).strip()
                if doc_content:
                    current_docs_content.append(doc_content)
                    loaded_titles.append(doc.get("title", "Untitled Document"))
            except Exception as e:
                print(f"Error loading document {doc_id}: {e}")
                continue

        if current_docs_content:
            vectorstore = build_vectorstore(current_docs_content)
            status_message = f"Successfully loaded {len(loaded_titles)} document(s): {', '.join(loaded_titles)}"
            return jsonify({"status": status_message})
        else:
            return jsonify({"status": "No documents could be loaded or documents are empty."}), 400

    except Exception as e:
        print(f"Error in load_doc: {e}")
        return jsonify({"status": f"Error loading documents: {str(e)}"}), 500

# RAG Chatbot
@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    data = request.json
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"answer": "Please enter a question."}), 400

    try:
        llm, retriever = build_qa_chain(vectorstore)

        if not retriever:
            fallback_llm = ChatOpenAI(
                model="provider-3/gpt-4o-mini",
                temperature=0.5,
                openai_api_key=A4F_API_KEY,
                openai_api_base=A4F_BASE_URL
            )
            response = fallback_llm.invoke(query)
            return jsonify({"answer": f"📚 Answer from general knowledge:\n\n{response.content}"})

        docs = retriever.get_relevant_documents(query)
        if not docs or not any(doc.page_content.strip() for doc in docs):
            response = llm.invoke(query)
            return jsonify({"answer": f"🔍 Answer not found in your documents.\nUsing general knowledge:\n\n{response.content}"})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain({"query": query})
        answer = f"📄 Answer based on your documents:\n\n{result['result']}"
        if result.get("source_documents"):
            answer += f"\n\n*Based on {len(result['source_documents'])} document section(s)*"

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return jsonify({"answer": "Sorry, there was an error processing your question."}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "vectorstore_loaded": vectorstore is not None})

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for("index"))

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    required_env_vars = ["SECRET_KEY", "HUGGINGFACEHUB_API_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    if not os.path.exists(CLIENT_SECRETS_FILE):
        print(f"Error: {CLIENT_SECRETS_FILE} not found.")
        exit(1)

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    print("🚀 Starting RAG Chatbot Server at http://localhost:5000")
    app.run(host="localhost", port=5000, debug=True)
