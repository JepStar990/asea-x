# **ASEA-X: Autonomous Software Engineering Agent System**

ASEA-X is a powerful **multi-agent AI system** that autonomously plans, develops, debugs, tests, and maintains software projects. It leverages LLMs (DeepSeek, OpenAI, Anthropics) and a modular architecture to provide an end‑to‑end software engineering workflow—from zero code to production-grade output.

***

## **📌 Features**

*   🤖 **Multi-Agent Architecture:** Planner, Developer, Linter, Debugger, Orchestrator
*   🧠 **LLM-Powered Intelligence:** DeepSeek API (default) with OpenAI fallback
*   🛡 **Safety First:** Multi-layer safety checks and override system
*   🧩 **Intelligent Planning:** Automatic task decomposition and architecture generation
*   🛠 **Autonomous Development:** Code writing, refactoring, testing, debugging
*   📚 **Documentation Enforcement:** Mandatory docstrings, JSDoc, and quality standards
*   🧪 **Error Recovery:** Runtime monitoring, test analysis, and self-fixing
*   🔍 **Context Awareness:** Semantic search via Vector DB
*   🗂 **State Management:** Redis support for distributed use
*   🎨 **Beautiful UI:** Rich TUI with real-time updates
*   🔧 **Git Integration:** Auto-commit with conventional commit messages

***

## **📖 System Overview**

ASEA-X is designed to:

*   Build projects **from scratch**
*   Identify and fix:
    *   Syntax errors
    *   Runtime errors
    *   Dependency issues
    *   Linting violations
*   Enforce:
    *   Documentation
    *   Code quality
    *   Safe operations
*   Support **complex project workflows** across multiple agents
*   Automatically select the next task or escalate when blocked
*   Provide a **human-friendly conversational interface**

***

## **🏗 Architecture**

ASEA-X uses a layered system:

### **🔹 Presentation Layer**

*   CLI
*   Terminal UI (textual)
*   REST API

### **🔹 Agent Layer**

*   **Orchestrator** – system brain
*   **Planner** – task decomposition & architecture
*   **Developer** – write/modify code, tests, dependencies
*   **Linter** – enforce code quality & documentation
*   **Debugger** – error classification & fix generation

### **🔹 Core Layer**

*   State Manager
*   Mode Manager
*   Safety System
*   LLM Manager

### **🔹 Execution Layer**

*   Command Executor
*   Runtime Monitor
*   File Manager
*   Git Manager

### **🔹 Data Layer**

*   Vector DB
*   Redis
*   File Context Store

***

## **🚀 Getting Started**

### **Requirements**

*   Python **3.10+** (3.11 recommended)
*   2 GB RAM minimum (8 GB recommended)
*   Linux, macOS, or Windows
*   Internet connection for LLM calls

***

### **Installation**

```bash
git clone https://github.com/JepStar990/asea-x.git
cd asea-x-main

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

***

### **Environment Configuration**

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

    DEEPSEEK_API_KEY=your_api_key_here
    DEEPSEEK_MODEL=deepseek-chat
    OPENAI_API_KEY=optional_openai_key
    AGENT_WORKDIR=./workdir
    SAFETY_ENABLED=true
    UI_THEME=dark

***

### **Start the system**

```bash
python -m src.main start
```

The terminal UI will launch.

***

## **💡 Usage**

ASEA-X operates through **modes**, controlled via slash commands.

### **Core Commands**

| Command          | Description                                |
| ---------------- | ------------------------------------------ |
| `/planner`       | Enter planning mode (architecture & tasks) |
| `/dev`           | Development mode (write/modify code)       |
| `/lint`          | Linting & documentation enforcement        |
| `/debug`         | Error investigation & fixing               |
| `/chat`          | Default conversational mode                |
| `/load <file>`   | Load file(s) into system context           |
| `/unsafe on/off` | Enable/disable safety override             |
| `/tasks`         | View task history                          |
| `/files`         | View active file context                   |
| `/status`        | Full system status                         |
| `/help`          | Display help                               |

***

### **Example Workflow**

1.  `/planner` → Describe what you want to build
2.  `/dev` → ASEA-X writes code, tests, and installs packages
3.  `/lint` → Ensure code meets quality standards
4.  `/debug` → Fix errors or failing tests
5.  `/dev` → Continue implementation

ASEA-X automatically transitions modes as needed.

***

## **🧭 Operational Modes**

| Mode          | Purpose                           | Trigger    |
| ------------- | --------------------------------- | ---------- |
| **Chat**      | Default conversation              | `/chat`    |
| **Planner**   | Task decomposition, architecture  | `/planner` |
| **Developer** | Coding, testing, execution        | `/dev`     |
| **Debug**     | Error analysis and self-fixing    | `/debug`   |
| **Lint**      | Enforce quality and documentation | `/lint`    |

***

## **🛡 Safety & Security**

ASEA-X is *safe by default*:

✔ Blocks dangerous commands (e.g., `rm -rf /`)  
✔ Prevents destructive file operations  
✔ Enforces resource limits  
✔ Monitors CPU/memory usage  
✔ Logs all unsafe override requests

Override via:

    /unsafe on

All overrides are logged with timestamp and reason.

***

## **🧰 Development & Contribution**

### **Code Standards**

*   Python code is checked with:
    *   Ruff
    *   Black
    *   Mypy
    *   Pydocstyle
*   JS/TS uses ESLint
*   Documentation must accompany all new code

### **Run tests**

```bash
pytest
```

### **Conventional Commits**

Example:

    feat(agent): add new planning heuristic

***

## **📜 License**

Licensed under the **MIT License**  
© 2026 Zwiswa Muridili  
See the `LICENSE` file for details.

***

## **❓ FAQ**

**Q: Can ASEA-X build a system from scratch?**  
Yes, it can generate architecture, code, tests, and docs—even with no initial files.

**Q: What happens when errors occur?**  
The Debugger Agent analyzes, proposes fixes, applies them, and validates the results.

**Q: Is this safe to run locally?**  
Yes. It includes strict safety layers and an explicit override system.

**Q: Does it support multiple languages?**  
Yes — Python, JavaScript, TypeScript, Java, Rust, Go, and more.

***

## **📚 Additional Documentation**

*   `DOCUMENTATION.md` — Deep technical system overview
*   `specification.md` — Formal system requirements
*   `runbook.md` — Setup and operational guide
