---
name: init-project
description: Initialize a new project with CLAUDE.md, directory structure, and optional configurations. Use when starting a new project or setting up Claude Code for an existing project.
disable-model-invocation: true
allowed-tools: Read, Write, Bash, AskUserQuestion
---

# Project Initialization Skill

Initialize a new project with standardized structure and Claude Code configuration.

## Usage

```bash
# Interactive mode (recommended)
/claude-init:init-project

# With arguments
/claude-init:init-project <project-name> <language> <project-type>
```

**Arguments:**
- `$ARGUMENTS[0]` - Project name (e.g., `my-app`)
- `$ARGUMENTS[1]` - Language: `python`, `typescript`, `general`
- `$ARGUMENTS[2]` - Project type: `api`, `cli`, `library`, `web`

## Workflow

### Step 1: Gather Information

If no arguments provided, ask the user:
1. Project name
2. Primary language (Python / TypeScript / General)
3. Project type (API / CLI / Library / Web App)
4. Optional features:
   - Initialize Git repository
   - Create language-specific config files

### Step 2: Create Directory Structure

Based on project type, create appropriate directories:

**API Project:**
```
<project>/
├── src/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── tests/
├── docs/
└── scripts/
```

**CLI Project:**
```
<project>/
├── src/
│   ├── commands/
│   ├── core/
│   └── utils/
├── tests/
└── docs/
```

**Library Project:**
```
<project>/
├── src/
│   └── <project_name>/
├── tests/
├── docs/
└── examples/
```

**Web App:**
```
<project>/
├── src/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   └── utils/
├── public/
├── tests/
└── docs/
```

### Step 3: Generate CLAUDE.md

Create `.claude/CLAUDE.md` with development conventions:

```markdown
# <Project Name>

## Development Conventions

### Task Planning
1. **Think Deeper First** - Break complex tasks into 3+ steps before starting
2. **Implementation Plan** - Write plans to `docs/implementation_plan.md` for complex tasks
3. **Incremental Development** - Minimum viable changes, keep code concise

### Code Style
- Clear naming for variables, functions, classes
- Single responsibility per function
- No code duplication
- Proper error handling

### Project Structure
<Insert generated structure here>
```

### Step 4: Language-Specific Setup (Optional)

**Python:**
- Create `pyproject.toml` with project metadata
- Create `requirements.txt`
- Add `.gitignore` for Python
- Copy `templates/rules/python.md` → `.claude/rules/python.md` (Schema-first Pydantic rules, module decoupling, error handling, logging conventions)

**TypeScript:**
- Create `package.json`
- Create `tsconfig.json`
- Add `.gitignore` for Node.js

### Step 5: Git Initialization (Optional)

If user opts in:
1. Run `git init`
2. Create initial commit: "Initial project setup"

## Output

After completion, display:
1. Summary of created files and directories
2. Next steps for the user
3. Available commands from this plugin

## Example Output

```
Project initialized successfully!

Created:
  .claude/CLAUDE.md
  .claude/rules/python.md
  src/api/
  src/schemas/
  src/services/
  tests/
  docs/
  .gitignore
  pyproject.toml

Next steps:
  1. cd my-app
  2. Create virtual environment: python -m venv .venv
  3. Start coding!

Available plugin commands:
  /claude-init:code-reviewer  - Review code quality
  /claude-init:test-writer    - Generate tests
  /claude-init:doc-writer     - Generate documentation

For Sprint-driven development:
  /multi-ralph:help        - See prerequisites and usage guide
  /multi-ralph:multi-ralph - Run Planner→Generator→Evaluator loop
```
