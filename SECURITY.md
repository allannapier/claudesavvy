# Security Policy

## Supported Versions

The following versions of Claude Monitor are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Data Privacy

Claude Monitor is designed with privacy in mind:

- **Local Only**: All data processing happens locally on your machine
- **No External Calls**: No data is sent to external servers or APIs
- **No Tracking**: No telemetry, analytics, or usage tracking
- **Read-Only**: Only reads from `~/.claude/` directory, never writes or modifies Claude Code data
- **No Credentials**: Does not require or store any API keys, passwords, or credentials

### What Data is Accessed

Claude Monitor reads the following local files:
- `~/.claude/history.jsonl` - Command history
- `~/.claude/projects/` - Session files with token usage
- `~/.claude/debug/` - MCP server activity logs
- `~/.claude/file-history/` - File modification tracking
- `~/.claude/skills/` - Installed skills
- `~/.claude/settings.json` - Claude Code settings

**All data stays on your machine.** The web server runs locally and by default listens on all network interfaces (0.0.0.0). For enhanced security, you can bind to localhost only using `--host 127.0.0.1`.

## Reporting a Vulnerability

If you discover a security vulnerability in Claude Monitor, please report it responsibly:

### How to Report

1. **GitHub Security Advisories** (Preferred):
   - Go to https://github.com/allannapier/claude_monitor/security/advisories
   - Click "Report a vulnerability"
   - Fill out the security advisory form

2. **Email** (Alternative):
   - Email details to the repository owner
   - Include "SECURITY" in the subject line
   - Provide detailed description of the vulnerability

### What to Include

Please include the following in your report:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)
- Your contact information (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next scheduled release

## Security Best Practices

When using Claude Monitor:

### For Users

1. **Network Access**: By default, the server binds to all network interfaces (0.0.0.0)
   - For enhanced security, bind to localhost only: `--host 127.0.0.1`
   - Never expose to the public internet without proper authentication
   - Use a firewall to restrict access when running on a network

2. **Keep Updated**: Use the latest version to get security fixes
   ```bash
   git pull
   pip install --upgrade .
   ```

3. **Review Dependencies**: All dependencies are from trusted sources
   - Flask: Web framework (official PyPI)
   - Jinja2: Template engine (official PyPI)
   - Click, Rich, python-dateutil: Standard Python libraries

4. **Data Location**: Never point `--claude-dir` to untrusted directories
   - Only use with your own Claude Code data
   - Default `~/.claude/` is safe

### For Developers

1. **Code Review**: All PRs require review before merging

2. **Dependencies**:
   - Pin dependency versions in requirements.txt
   - Regular dependency updates for security patches
   - No external API calls or network requests

3. **Input Validation**:
   - All user inputs are validated
   - Path traversal protection on file reads
   - No execution of user-provided code

4. **Template Security**:
   - Jinja2 auto-escaping enabled by default
   - No use of `| safe` filter without careful review
   - All dynamic content properly escaped

## Known Limitations

Claude Monitor is a development/monitoring tool and has intentional limitations:

1. **No Authentication**: The web interface has no built-in authentication
   - Designed for local use only
   - If you need multi-user access, add authentication proxy (e.g., nginx)

2. **No HTTPS**: Web server uses HTTP, not HTTPS
   - Safe for localhost use
   - If exposing to network, use reverse proxy with TLS

3. **Debug Mode**: `--debug` flag should never be used in production
   - Exposes internal errors and stack traces
   - Enables auto-reload which could cause instability

## Disclosure Policy

When a security issue is reported:

1. **Acknowledgment**: We'll acknowledge receipt within 48 hours
2. **Investigation**: We'll investigate and validate the issue
3. **Fix Development**: We'll develop and test a fix
4. **Coordinated Disclosure**: We'll work with the reporter on disclosure timeline
5. **Public Disclosure**: After fix is released, we'll publish:
   - Security advisory on GitHub
   - Details in CHANGELOG.md
   - Credit to the reporter (if desired)

## Security Updates

Security fixes are released as patch versions (e.g., 1.0.1) and announced via:
- GitHub Security Advisories
- Release notes
- CHANGELOG.md

Subscribe to repository notifications to stay informed.

## Contact

For security concerns that don't require immediate attention:
- Open an issue on GitHub (for non-sensitive matters)
- Use GitHub Discussions for questions

For sensitive security matters:
- Use GitHub Security Advisories (preferred)
- Or contact the repository owner directly

---

**Last Updated**: 2025-12-20
