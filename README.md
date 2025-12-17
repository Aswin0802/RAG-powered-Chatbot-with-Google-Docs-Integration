

# 📄 Google Docs RAG Chatbot

A Flask-based AI assistant that allows you to securely log in with your Google account, ingest your Google Documents, and chat with them using Retrieval Augmented Generation (RAG).



## 🚀 Features

- **OAuth2 Authentication**: Secure login via Google to access private Drive files.
- **Document Indexing**: Fetches text directly from Google Docs API.
- **Vector Search**: Uses FAISS and HuggingFace embeddings (`all-mpnet-base-v2`) for semantic search.
- **Contextual Chat**: Uses an OpenAI-compatible LLM to answer questions based strictly on your document content.
- **Source Citing**: The bot indicates when it is using document context vs. general knowledge.

---

## 🛠️ Tech Stack

- **Backend**: Flask
- **Orchestration**: LangChain
- **Embeddings**: HuggingFace Endpoint
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Auth**: Google OAuthLib

---

## ⚙️ Prerequisites

1. **Python 3.9+**
2. **Google Cloud Project** with:
   - Google Drive API enabled
   - Google Docs API enabled
3. **API Keys** for HuggingFace and your LLM provider.

---

## 📝 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/google-docs-rag.git](https://github.com/yourusername/google-docs-rag.git)

```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install flask python-dotenv google-auth-oauthlib google-api-python-client langchain-community langchain-huggingface langchain-openai faiss-cpu

```

---

## 🔐 Configuration

### 1. Google OAuth Setup (Crucial)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a Project and enable **Google Drive API** and **Google Docs API**.
3. Go to **Credentials** → **Create Credentials** → **OAuth Client ID**.
4. **Application Type**: Web Application.
5. **Authorized Redirect URIs**: Add `http://localhost:5000/callback`
6. Download the JSON file, rename it to `credentials.json`, and place it in the root folder.

### 2. Environment Variables

Create a `.env` file in the root directory:

```ini
SECRET_KEY=your_random_secret_string_here
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token

# LLM Configuration (OpenAI Compatible)
A4F_API_KEY=your_llm_api_key
A4F_BASE_URL=[https://api.your-provider.com/v1](https://api.your-provider.com/v1)

```

---

## ▶️ Usage

1. **Start the Server**:
```bash
python app.py

```


2. **Open Browser**: Navigate to `http://localhost:5000`.
3. **Login**: Click login to authenticate with Google.
4. **Load Docs**: Select documents from the list and click "Load" to index them.
5. **Ask**: Type your question to chat with the content of the loaded documents.

---

## ⚠️ Important Notes

* **Localhost Only**: The app sets `OAUTHLIB_INSECURE_TRANSPORT=1` to allow OAuth over HTTP. **Do not use this setting in production**; production requires HTTPS.
* **Memory Storage**: The vector store is rebuilt in memory every time the application restarts. It does not persist to disk in this version.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
