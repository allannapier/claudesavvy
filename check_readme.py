#!/usr/bin/env python3
"""Check if README.md is up to date with the project.

This script performs several checks to ensure README.md stays synchronized
with the project's configuration and codebase:

1. Version Consistency: Verifies that pyproject.toml, setup.py, and 
   CHANGELOG.md all have matching version numbers.

2. Version Documentation: Ensures the README mentions the current version
   and includes recent versions from the CHANGELOG.

3. Directory Paths: Checks for outdated directory names (like 'claude_monitor'
   instead of 'claudesavvy').

4. Project Structure: Validates that key directories referenced in the README
   actually exist.

Usage:
    python check_readme.py

Exit codes:
    0: All checks passed, README is up to date
    1: One or more checks failed, README needs updating

This script is run automatically in CI to catch inconsistencies.
"""

import re
import sys
from pathlib import Path


def extract_version_from_file(filepath, pattern):
    """Extract version string from a file using regex pattern."""
    try:
        content = Path(filepath).read_text()
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None


def check_versions():
    """Check version consistency across files."""
    errors = []
    
    # Extract versions
    pyproject_version = extract_version_from_file(
        "pyproject.toml",
        r'version\s*=\s*"([^"]+)"'
    )
    setup_version = extract_version_from_file(
        "setup.py",
        r'version\s*=\s*"([^"]+)"'
    )
    
    # Get latest version from CHANGELOG
    changelog_content = Path("CHANGELOG.md").read_text()
    changelog_match = re.search(r'##\s+\[([^\]]+)\]', changelog_content)
    changelog_version = changelog_match.group(1) if changelog_match else None
    
    print("Version Information:")
    print(f"  pyproject.toml: {pyproject_version}")
    print(f"  setup.py:       {setup_version}")
    print(f"  CHANGELOG.md:   {changelog_version}")
    
    # Check consistency
    if pyproject_version != setup_version:
        errors.append(
            f"Version mismatch: pyproject.toml ({pyproject_version}) != "
            f"setup.py ({setup_version})"
        )
    
    # Check if versions match CHANGELOG
    if changelog_version and pyproject_version != changelog_version:
        errors.append(
            f"Version in pyproject.toml ({pyproject_version}) does not match "
            f"latest CHANGELOG version ({changelog_version})"
        )
    
    # Check README mentions current version
    readme_content = Path("README.md").read_text()
    if pyproject_version and f"v{pyproject_version}" not in readme_content:
        errors.append(
            f"README.md does not mention current version v{pyproject_version}"
        )
    
    return errors


def check_directory_paths():
    """Check for incorrect directory paths in README."""
    errors = []
    readme_content = Path("README.md").read_text()
    
    # Check for old directory name
    if "cd claude_monitor" in readme_content:
        errors.append(
            "README.md contains incorrect directory path 'claude_monitor' "
            "instead of 'claudesavvy'"
        )
    
    # Check for old package references in examples
    if "claude-monitor" in readme_content.lower():
        # Filter out historical references (like in version history or migration guides)
        lines_with_old_name = []
        for i, line in enumerate(readme_content.split('\n'), 1):
            if "claude-monitor" in line.lower() and not any(x in line for x in [
                "Migration", "Before", "v1.x", "upgrade", "uninstall"
            ]):
                lines_with_old_name.append(f"line {i}: {line.strip()}")
        
        if lines_with_old_name:
            errors.append(
                f"README.md contains old package name 'claude-monitor' in "
                f"active content: {', '.join(lines_with_old_name[:3])}"
            )
    
    return errors


def check_project_structure():
    """Check if project structure in README matches actual structure."""
    errors = []
    readme_content = Path("README.md").read_text()
    
    # Extract project structure section from README
    structure_match = re.search(
        r'```\nclaude_monitor/(.*?)```',
        readme_content,
        re.DOTALL
    )
    
    if structure_match and "claude_monitor/" in structure_match.group(0):
        errors.append(
            "README.md project structure section uses 'claude_monitor/' "
            "instead of 'claudesavvy/'"
        )
    
    # Check key directories exist
    key_dirs = [
        "src/claudesavvy",
        "src/claudesavvy/web",
        "src/claudesavvy/parsers",
        "src/claudesavvy/analyzers",
    ]
    
    missing_dirs = []
    for dir_path in key_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        errors.append(f"Expected directories not found: {', '.join(missing_dirs)}")
    
    return errors


def check_changelog_in_readme():
    """Check that recent CHANGELOG entries are reflected in README."""
    errors = []
    
    changelog_content = Path("CHANGELOG.md").read_text()
    readme_content = Path("README.md").read_text()
    
    # Extract top 3 versions from CHANGELOG
    version_pattern = r'##\s+\[([^\]]+)\]'
    versions = re.findall(version_pattern, changelog_content)[:3]
    
    # Check if these versions are in README version history
    for version in versions:
        if f"v{version}" not in readme_content:
            errors.append(
                f"Recent version v{version} from CHANGELOG is not in README version history"
            )
    
    return errors


def main():
    """Run all README checks."""
    print("Checking README.md consistency...\n")
    
    all_errors = []
    
    # Run all checks
    all_errors.extend(check_versions())
    all_errors.extend(check_directory_paths())
    all_errors.extend(check_project_structure())
    all_errors.extend(check_changelog_in_readme())
    
    # Report results
    print()
    if all_errors:
        print("❌ README.md is NOT up to date:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        return 1
    else:
        print("✅ README.md is up to date!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
