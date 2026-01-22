# **ASEA-X: Autonomous Software Engineering Agent System**

---

## **0. System Guarantees (Non-Negotiable)**

ASEA-X **must** be able to:

1. Start from **zero code** and produce a **running system**
2. Detect and fix:

   * Syntax errors
   * Runtime errors
   * Dependency issues
   * Lint violations
3. Enforce:

   * Linting
   * Documentation
4. Operate safely by default, with **explicit override**
5. Coordinate **multiple agents**
6. Autonomously suggest and progress to **next tasks**
7. Operate through a **human-friendly UI**

If any of these fail, the system halts and escalates.

---

## **1. System Architecture Overview**

### **1.1 High-Level Components**

```
┌─────────────────────────────┐
│        User Interface       │
│  (Pretty Chat + Command UI) │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│   Orchestrator Agent (OA)   │  ← DeepSeek-powered
│  (Mode Switch + Routing)   │
└───────┬──────────┬──────────┘
        │          │
┌───────▼───┐  ┌───▼─────────┐
│ Planner   │  │ Dev Agent   │
│ Agent     │  │ (Executor)  │
└───────┬───┘  └───┬─────────┘
        │          │
   ┌────▼────┐  ┌──▼─────────┐
   │ Lint &  │  │ Debug Agent │
   │ Docs    │  │             │
   └─────────┘  └─────────────┘
```

---

## **2. Language Model Backend**

### **2.1 LLM Provider**

* **DeepSeek API**
* Supports:

  * Long context
  * Code-first reasoning
  * Tool calling

### **2.2 Model Usage Rules**

* Orchestrator decides:

  * Which agent to invoke
  * Which mode to activate
* Agents never self-elevate privileges

---

## **3. Agent Types & Responsibilities**

---

### **3.1 Orchestrator Agent (OA)**

**Role:** System brain

#### Responsibilities

* Interpret user intent
* Switch modes automatically
* Dispatch tasks to sub-agents
* Maintain global system state
* Decide when to:

  * Continue
  * Suggest next steps
  * Halt and escalate

#### Mode Switching

OA **may override user mode** if:

* A task requires execution
* Planning is incomplete
* Debugging is required

---

### **3.2 Planner Agent**

**Role:** Architecture & task decomposition

#### Responsibilities

* Translate ideas → atomic tasks
* Define file/module boundaries
* Identify dependencies
* Define acceptance criteria

#### Output

* Task graph
* Ordered checklist
* No code

---

### **3.3 Dev Agent**

**Role:** Code execution engine

#### Responsibilities

* Write code
* Modify files
* Run commands
* Install dependencies (with permission)
* Commit changes

---

### **3.4 Lint & Documentation Agent**

**Role:** Quality gatekeeper

#### Responsibilities

* Enforce linting
* Enforce documentation
* Reject incomplete code

---

### **3.5 Debug Agent**

**Role:** Failure recovery

#### Responsibilities

* Analyze runtime output
* Classify errors
* Patch minimal fixes
* Re-run execution loop

---

## **4. Operational Modes (System-Enforced)**

| Mode       | Trigger             | Description         |
| ---------- | ------------------- | ------------------- |
| `/chat`    | Default             | Discussion only     |
| `/planner` | Architecture needed | Task breakdown      |
| `/dev`     | Code needed         | Execution           |
| `/debug`   | Error detected      | Runtime recovery    |
| `/lint`    | Pre-commit          | Quality enforcement |

---

## **5. File Loading & Context Injection**

### **5.1 User-Loaded Files**

Users may explicitly load files into the system:

```
/load ./specs/api.md
/load ./src/main.py
```

#### Behavior

* Files are:

  * Parsed
  * Summarized
  * Indexed
* Stored in the local vector DB
* Used as authoritative context

---

## **6. Linting & Documentation (Mandatory)**

### **6.1 Universal Requirement**

**NO CODE MAY BE COMMITTED UNLESS:**

1. Lint passes
2. Documentation exists

---

### **6.2 Language-Specific Enforcement**

| Language | Linter        | Docs                 |
| -------- | ------------- | -------------------- |
| Python   | ruff / flake8 | docstrings (PEP 257) |
| JS/TS    | eslint        | JSDoc                |
| Java     | checkstyle    | Javadoc              |
| Rust     | clippy        | rustdoc              |
| Go       | golangci-lint | GoDoc                |
| Shell    | shellcheck    | comments             |

---

### **6.3 Enforcement Flow**

```
write code
→ run linter
→ fix violations
→ verify docs
→ commit
```

Failure → automatic fix loop or escalation.

---

## **7. Dependency & Installation System**

### **7.1 Permission Model**

* All installs require explicit user approval
* System detects missing dependencies automatically

### **7.2 Verification Loop**

* Install
* Verify import/version
* Retry (max 3)
* Escalate

---

## **8. Runtime Execution & Debugging**

### **8.1 Execution Observer**

Captures:

* stdout
* stderr
* exit code

### **8.2 Error Classification**

| Type              | Action         |
| ----------------- | -------------- |
| ImportError       | Install flow   |
| Test failure      | Debug Agent    |
| Runtime exception | Patch          |
| Segfault          | Halt           |
| Infinite loop     | Kill + analyze |

---

## **9. Environment Safety System**

### **9.1 Default State**

✅ **ON by default**

Includes:

* Virtual environments
* Command safety filters
* No system-wide installs
* No destructive commands

---

### **9.2 Override Mechanism**

```
/unsafe on
/unsafe off
```

#### Constraints

* Requires confirmation
* Logged explicitly
* Auto-reverts after session unless persisted

---

## **10. Git & Version Control**

* Atomic commits only
* Conventional commit messages
* Shadow branches for complex tasks
* Mandatory verification before commit

---

## **11. Task Progression & Autonomy**

### **11.1 State Evaluation**

After each task, OA evaluates:

* Git status
* Test results
* File completeness
* TODO markers

---

### **11.2 Next-Step Suggestion**

Example:

> ✅ API implemented
> ⏭ Suggested next step:
>
> * Add input validation
> * Write unit tests
> * Generate README

OA may automatically proceed **unless user blocks**.

---

## **12. UI / UX Specification**

### **12.1 Pretty Chat Interface**

Requirements:

* Rich Markdown rendering
* Code blocks
* Diff highlighting
* Tables

---

### **12.2 Input Handling (Critical)**

#### Problem

Enter key should **NOT** trigger execution.

#### Solution

* `Enter` → New line
* `Ctrl + Enter` → Submit
* Explicit **Run / Execute** button

This allows:

* Copy/paste
* Editing
* Multi-line commands

---

## **13. Logging & Auditability**

| Log File              | Purpose            |
| --------------------- | ------------------ |
| `.agent.actions.log`  | Decisions          |
| `.agent.errors.log`   | Failures           |
| `.agent.installs.log` | Dependency changes |
| `.agent.safety.log`   | Unsafe overrides   |

---

## **14. Can It Build From Scratch?**

**Yes. By design.**

Because:

* Planner decomposes idea
* Dev writes full code
* Lint enforces correctness
* Debug loop fixes runtime errors
* Dependency manager resolves environment
* OA coordinates progression

If it cannot proceed:
➡️ It **must explain exactly why**.

---

## **15. Final System Properties**

| Property           | Status |
| ------------------ | ------ |
| Autonomous         | ✅      |
| Safe by default    | ✅      |
| Overrideable       | ✅      |
| Multi-agent        | ✅      |
| End-to-end capable | ✅      |
| Production-grade   | ✅      |
