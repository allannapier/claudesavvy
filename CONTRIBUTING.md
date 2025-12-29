# Contributing to ClaudeSavvy

Thank you for your interest in contributing to ClaudeSavvy! This document provides guidelines and instructions for contributing to the project.

## Welcome

We welcome contributions of all kinds:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Performance optimizations
- Code quality improvements

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to build something useful together.

## Getting Started

### 1. Clone and Set Up

```bash
# Clone the repository
git clone https://github.com/allannapier/claude_monitor.git
cd claude_monitor

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with requirements
pip install -r requirements.txt
```

### 2. Run Locally

```bash
# Start the web server
claudesavvy

# Or run directly
python3 -m src.claudesavvy.cli

# Enable debug mode for development
claudesavvy --debug
```

The web interface will be available at http://localhost:5000

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring

### Commit Messages

Write clear, descriptive commit messages:

```
Add export functionality for CSV format

- Implement CSV export route
- Add download headers
- Include all dashboard metrics
```

### Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** with clear, focused commits

3. **Test your changes**:
   - Verify the web server starts successfully
   - Test all affected routes and functionality
   - Check for Python errors and warnings

4. **Update documentation** if needed:
   - Update README.md for user-facing changes
   - Add docstrings for new functions/classes
   - Update CHANGELOG.md with your changes

5. **Submit a pull request**:
   - Describe what changes you made and why
   - Link to any related issues
   - Include screenshots for UI changes

## Project Structure

Understanding the architecture will help you contribute effectively:

```
src/claudesavvy/
├── cli.py                 # Entry point - launches web server
├── parsers/              # Data parsers (read Claude Code files)
│   ├── history.py        # Command history parser
│   ├── sessions.py       # Session and token data parser
│   ├── debug.py          # MCP server logs parser
│   ├── files.py          # File editing history parser
│   ├── tools.py          # Tool usage parser
│   └── skills.py         # Skills and config parser
├── analyzers/            # Data analysis layer
│   ├── usage.py          # Usage statistics analyzer
│   ├── tokens.py         # Token usage and cost analyzer
│   ├── integrations.py   # MCP integration analyzer
│   └── features.py       # Features analyzer
├── web/                  # Flask web application
│   ├── app.py            # Flask app factory
│   ├── routes/           # Route handlers
│   │   └── dashboard.py  # Main dashboard routes
│   ├── services/         # Business logic
│   │   └── dashboard_service.py
│   ├── templates/        # Jinja2 HTML templates
│   │   ├── pages/        # Full page templates
│   │   ├── partials/     # Reusable components
│   │   └── components/   # UI components
│   └── static/           # CSS, JS, images
└── utils/                # Shared utilities
    ├── paths.py          # Path management
    └── time_filter.py    # Time-based filtering
```

## Adding Features

### Adding a New Metric

1. **Create/modify a parser** in `parsers/` to extract the data
2. **Create/modify an analyzer** in `analyzers/` to process the data
3. **Add to web service** in `web/services/dashboard_service.py`
4. **Create a route** in `web/routes/dashboard.py`
5. **Create a template** in `web/templates/pages/`
6. **Update navigation** in `web/templates/base.html`

### Adding a New Parser

Parsers read Claude Code data files:

```python
from pathlib import Path
from typing import List

class MyParser:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def parse(self) -> List[dict]:
        """Parse the file and return structured data."""
        # Your parsing logic here
        pass
```

### Adding a New Analyzer

Analyzers process parsed data:

```python
from .parsers.my_parser import MyParser
from .utils.time_filter import TimeFilter

class MyAnalyzer:
    def __init__(self, parser: MyParser, time_filter: TimeFilter):
        self.parser = parser
        self.time_filter = time_filter

    def get_summary(self):
        """Analyze data and return summary."""
        # Your analysis logic here
        pass
```

### Adding a New Web Page

1. **Add route** in `web/routes/dashboard.py`:
   ```python
   @dashboard_bp.route('/my-page')
   def my_page():
       data = dashboard_service.get_my_data()
       return render_template('pages/my_page.html', data=data)
   ```

2. **Create template** in `web/templates/pages/my_page.html`:
   ```html
   {% extends "base.html" %}
   {% block content %}
   <!-- Your page content -->
   {% endblock %}
   ```

## Testing

Currently, testing is manual. When making changes:

### Manual Testing Checklist

- [ ] Web server starts without errors
- [ ] All existing pages load correctly
- [ ] New functionality works as expected
- [ ] No Python exceptions in console
- [ ] Data displays correctly with real Claude data
- [ ] Export functionality works (if modified)
- [ ] Browser console has no errors
- [ ] Responsive design works (if UI changes)

### Test in Multiple Scenarios

- Test with minimal Claude usage data
- Test with extensive Claude usage data
- Test with missing/incomplete data files
- Test edge cases (no sessions, no projects, etc.)

## Style Guide

### Python Style

- Follow PEP 8 conventions
- Use type hints where helpful:
  ```python
  def calculate_cost(tokens: int) -> float:
      return tokens / 1_000_000 * 3.0
  ```
- Write descriptive variable names
- Add docstrings for public functions:
  ```python
  def get_summary(self) -> dict:
      """
      Get usage summary statistics.

      Returns:
          dict: Summary statistics including sessions, commands, and projects
      """
  ```

### HTML/Jinja2 Style

- Use semantic HTML elements
- Keep templates DRY (Don't Repeat Yourself)
- Use partials for reusable components
- Include helpful comments in complex templates

### Code Organization

- Keep functions focused and single-purpose
- Avoid deep nesting (max 3-4 levels)
- Extract complex logic into helper functions
- Group related functionality together

## Submitting Pull Requests

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All manual tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No commented-out code or debug statements
- [ ] No secrets or personal data in code

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Changes Made
- Bullet point list of changes

## Testing Done
- How you tested these changes

## Screenshots (if UI changes)
[Add screenshots here]

## Related Issues
Closes #123
```

## Questions or Issues?

- Open an issue on GitHub for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## License

By contributing to ClaudeSavvy, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ClaudeSavvy! 🚀
