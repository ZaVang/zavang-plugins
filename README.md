# ZaVang Claude Plugins

A Claude Code plugin marketplace for project initialization, code review, testing, and more.

## Available Plugins

### claude-autoupdate
One-click Claude Code upgrade with automatic changelog summarization. Detects your current version, upgrades to latest, fetches every release note in between, and produces a structured report with deep-dive explanations of core new features.

**Features:**
- **One-click Upgrade**: Detects current version, fetches latest from npm, and upgrades in one command
- **Version-by-Version Changelog**: Summarizes every release between old and new version
- **Core Feature Deep-Dives**: Detailed explanations (What / Why / How / Impact) for major features
- **Saved Report**: Writes a Markdown report to `~/.claude/` for future reference
- **Smart Fallbacks**: Falls back to web search if GitHub releases are unavailable

### claude-init
One-click project initialization with CLAUDE.md templates, directory structure, and built-in agents.

**Features:**
- **Project Initialization**: Generate CLAUDE.md, directory structure, and language-specific configurations
- **Built-in Agents**: Code review, test generation, documentation, architecture analysis, debugging
- **Auto-formatting Hook**: Python files auto-formatted with black + isort
- **Templates**: Customizable project templates for different project types

### llm-bridge
Drop a production-ready LLM gateway module into any Python project with a single command. Unifies OpenAI, Anthropic, and Google Gemini/Vertex AI behind one clean interface.

**Features:**
- **Unified `LLMBridge.chat()` API**: one call works across all providers
- **Pydantic Structured Output**: pass a `response_model` and get a typed object back
- **Retry with Exponential Backoff**: configurable attempts, delays, and jitter
- **Dynamic Fallback**: if model A fails, automatically try model B
- **API Key Rotation**: round-robin across multiple keys per provider
- **Flexible Placement**: choose any parent dir (`src/services/`, `src/`, `infra/`, …) and any module name (`llm_bridge`, `ai_client`, `llm_gateway`, …)
- **Guard Hooks**: warns if direct SDK imports or hardcoded API keys appear in project files

### perspective-encoder
A modular multi-perspective reasoning framework for the deepthought factor prediction system. Designed to integrate LLM-based qualitative analysis with Bayesian Network quantitative inference.

**Features:**
- **Four-Layer Perspective Architecture**:
  - Layer 0: Exogenous variable inference (RegimeState, EconCycle, MonetaryCycle)
  - Layer 1: Transmission path perspectives (policy, liquidity, cycle)
  - Layer 2: Regime-specific specialists (risk_on, risk_off, crisis, recovery)
  - Layer 3: Event-type analysts (monetary_policy, geopolitical, trade)
  - Layer 4: Mediator-specific analysts (10 specialized analysts)
- **Dynamic Prompt Assembly**: Automatically selects and combines perspectives based on market state and event type
- **Market State Tracking**: Persistent tracking with transition history
- **Mediator Analysis**: Deep analysis for each of the 10 mediators in the causal network

**Use Cases:**
- Factor prediction for quantitative trading systems
- Market regime classification and state tracking
- Multi-agent investment analysis pipelines

## Installation

### Option 1: Plugin Marketplace (Recommended)

```bash
# Add this repo as a marketplace
/plugin marketplace add ZaVang/zavang-plugins

# Install the plugin
/plugin install claude-init@zavang-plugins
```

### Option 2: Clone to plugins directory

```bash
# Clone the entire marketplace
git clone https://github.com/ZaVang/zavang-plugins.git ~/.claude/plugins/zavang-plugins

# Restart Claude Code to load the plugins
```

### Option 3: Using --plugin-dir (For development)

```bash
# Clone the repository
git clone https://github.com/ZaVang/zavang-plugins.git

# Start Claude Code with the marketplace
claude --plugin-dir /path/to/claude-init
```

## Quick Start

### Initialize a New Project

```bash
# Interactive mode (recommended)
/claude-init:init-project

# With arguments: <project-name> <language> <project-type>
/claude-init:init-project myapp python api
```

**Supported Languages**: `python`, `typescript`, `general`

**Project Types**: `api`, `cli`, `library`, `web`

### What Gets Created

```
myapp/
├── .claude/
│   ├── CLAUDE.md           # Development conventions
│   └── rules/
│       └── python.md       # Language-specific rules
├── src/
│   ├── api/                # API endpoints
│   ├── schemas/            # Data models (Pydantic)
│   ├── services/           # Business logic
│   └── utils/              # Utilities
├── tests/                  # Test files
├── docs/                   # Documentation
├── .gitignore
└── pyproject.toml          # Python config
```

### Available Commands

### claude-autoupdate

| Command | Description |
|---------|-------------|
| `/claude-autoupdate:autoupdate` | Upgrade Claude Code to latest version and generate a full changelog report |

### claude-init

| Command | Description |
|---------|-------------|
| `/claude-init:init-project` | Initialize a new project with structure and configs |
| `/claude-init:code-reviewer` | Review code for quality, security, and performance |
| `/claude-init:test-writer` | Generate pytest test cases |
| `/claude-init:doc-writer` | Generate documentation and docstrings |
| `/claude-init:architect` | Analyze project architecture |
| `/claude-init:debugger` | Debug errors and trace issues |

### llm-bridge

| Command | Description |
|---------|-------------|
| `/llm-bridge:llm-bridge-python` | Add a production-ready LLM Bridge module to any Python project |

## Agents

### Code Reviewer
Reviews code for:
- Code quality (naming, single responsibility, duplication)
- Security issues (SQL injection, hardcoded secrets)
- Performance (N+1 queries, memory leaks)
- Python best practices (type hints, error handling)

### Test Writer
Generates pytest tests following:
- AAA pattern (Arrange, Act, Assert)
- Coverage for happy path, edge cases, exceptions
- Proper use of fixtures and mocks

### Doc Writer
Creates documentation:
- Google-style docstrings
- Module-level READMEs
- API documentation

### Architect
Analyzes:
- Project structure and organization
- Code architecture and layering
- Design patterns and anti-patterns
- Extensibility and maintainability

### Debugger
Helps with:
- Error stack trace analysis
- Root cause identification
- Fix suggestions with prevention tips

## Hooks

### Python Auto-formatting
Automatically formats Python files on save:
- **black**: Code formatting
- **isort**: Import sorting

## Requirements

For Python formatting hook:
```bash
pip install black isort
```

## Marketplace Structure

```
zavang-plugins/
├── plugins/
│   ├── claude-autoupdate/      # claude-autoupdate plugin
│   │   ├── skills/
│   │   │   └── autoupdate/
│   │   │       └── SKILL.md
│   │   └── agents/
│   │       └── changelog-analyst/
│   │           └── AGENT.md
│   ├── claude-init/            # claude-init plugin
│   │   ├── skills/
│   │   │   └── init-project/
│   │   │       └── SKILL.md
│   │   ├── agents/
│   │   │   ├── code-reviewer/
│   │   │   ├── test-writer/
│   │   │   ├── doc-writer/
│   │   │   ├── architect/
│   │   │   └── debugger/
│   │   └── hooks/
│   │       └── hooks.json
│   └── llm-bridge/             # llm-bridge plugin
│       ├── .claude-plugin/
│       │   └── plugin.json
│       ├── skills/
│       │   └── llm-bridge-python/
│       │       ├── SKILL.md
│       │       └── templates/
│       │           ├── llm_bridge/      # module source template
│       │           ├── llm_bridge_config.example.yaml
│       │           └── test_*.py
│       └── hooks/
│           └── hooks.json
└── README.md
```

## Development

### Local Testing

```bash
claude --plugin-dir /path/to/claude-init
```

### Contributing

1. Fork this repository
2. Create your feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

[ZaVang](https://github.com/ZaVang)
