# 🤖 AgentCoder AI

An intelligent code generation and analysis platform powered by **Pydantic AI** agents and **Streamlit**.

Run it here: https://codeaiagent.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

### 💻 Code Generation
- Generate production-ready code from natural language descriptions
- Support for multiple programming languages (Python, JavaScript, TypeScript, Java, Go, Rust)
- Automatic language detection
- Optional unit test generation
- Comprehensive documentation included

### 🔍 Code Analysis
- Comprehensive code review with quality scoring (0-100)
- Security vulnerability detection
- Performance optimization suggestions
- Best practices evaluation
- Refactored code suggestions

### 📚 Documentation RAG
- Upload PDF documentation
- Automatic text extraction and chunking
- Vector-based semantic search
- Chat interface for Q&A
- Source citations with page numbers

### 📋 Agent Workflow Logs
- Real-time visibility into agent reasoning
- ReAct-style thought/action/observation steps
- Full transparency into AI decision-making

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentcoder-ai.git
cd agentcoder-ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

5. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
agentcoder-ai/
├── app.py                      # Main Streamlit application
├── agents/
│   ├── code_agent.py          # Code generation agent
│   ├── analysis_agent.py      # Code analysis agent
│   └── rag_agent.py           # Documentation RAG agent
├── tools/
│   ├── document_parser.py     # PDF parsing utilities
│   ├── code_tools.py          # Code processing utilities
│   └── retrieval.py           # Vector search tools
├── models/
│   └── schemas.py             # Pydantic data models
├── utils/
│   ├── embeddings.py          # Sentence transformer embeddings
│   └── storage.py             # FAISS vector store & document storage
├── data/
│   └── uploaded_docs/         # User uploaded PDFs
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Required |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `DEFAULT_MODEL` | Default LLM model | `anthropic/claude-3.5-sonnet` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | PDF chunk size (chars) | `500` |
| `CHUNK_OVERLAP` | Chunk overlap (chars) | `50` |

### Supported Models

- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4-turbo`
- `openai/gpt-4o`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-70b-instruct`

## ☁️ Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in the Streamlit dashboard:
   ```toml
   OPENROUTER_API_KEY = "your-api-key-here"
   ```
5. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black . && isort .
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

Built with ❤️ using Pydantic AI and Streamlit
