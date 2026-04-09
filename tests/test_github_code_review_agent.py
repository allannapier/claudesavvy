"""Tests for the GitHub Code Review Agent."""

from claudesavvy.agents.github_code_review_agent import (
    PullRequest, ReviewComment, GitHubCodeReviewAgent
)


def test_pull_request_from_github_api():
    """Test creating PullRequest from GitHub API data."""
    api_data = {
        'number': 68,
        'title': 'Add GitHub Issue Agent',
        'body': 'This PR adds a new agent',
        'state': 'open',
        'author': {'login': 'Claude'},
        'baseRefName': 'main',
        'headRefName': 'feature/issue-agent',
        'createdAt': '2024-01-01T00:00:00Z',
        'updatedAt': '2024-01-02T00:00:00Z',
        'url': 'https://github.com/test/test/pull/68',
        'additions': 1069,
        'deletions': 0,
        'changedFiles': 10
    }

    pr = PullRequest.from_github_api(api_data)

    assert pr.number == 68
    assert pr.title == 'Add GitHub Issue Agent'
    assert pr.body == 'This PR adds a new agent'
    assert pr.state == 'open'
    assert pr.author == 'Claude'
    assert pr.base_branch == 'main'
    assert pr.head_branch == 'feature/issue-agent'
    assert pr.url == 'https://github.com/test/test/pull/68'
    assert pr.additions == 1069
    assert pr.deletions == 0
    assert pr.changed_files == 10


def test_analyze_security_issues_hardcoded_secrets():
    """Test detection of hardcoded secrets."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/config.py b/config.py
index 1234567..abcdefg 100644
--- a/config.py
+++ b/config.py
@@ -1,3 +1,4 @@
 DATABASE_URL = 'postgresql://localhost/db'
+password = 'supersecret123'
 API_ENDPOINT = 'https://api.example.com'
"""

    comments = agent.analyze_security_issues(diff)

    assert len(comments) > 0
    assert any('hardcoded secret' in c.comment.lower() for c in comments)
    assert any(c.severity == 'error' for c in comments)


def test_analyze_security_issues_sql_injection():
    """Test detection of SQL injection vulnerabilities."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/database.py b/database.py
index 1234567..abcdefg 100644
--- a/database.py
+++ b/database.py
@@ -10,3 +10,4 @@ def get_user(user_id):
     return db.query("SELECT * FROM users WHERE id = ?", user_id)
+def search_users(query):
+    return db.execute(f"SELECT * FROM users WHERE name LIKE '{query}'")
"""

    comments = agent.analyze_security_issues(diff)

    assert len(comments) > 0
    assert any('sql injection' in c.comment.lower() for c in comments)
    assert any(c.severity == 'error' for c in comments)


def test_analyze_security_issues_eval_usage():
    """Test detection of eval/exec usage."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/utils.py b/utils.py
index 1234567..abcdefg 100644
--- a/utils.py
+++ b/utils.py
@@ -5,3 +5,5 @@ def process_input(data):
     return json.loads(data)
+def execute_code(code):
+    return eval(code)
"""

    comments = agent.analyze_security_issues(diff)

    assert len(comments) > 0
    assert any('eval' in c.comment.lower() or 'injection' in c.comment.lower() for c in comments)
    assert any(c.severity == 'warning' for c in comments)


def test_analyze_security_issues_log_injection():
    """Test detection of log injection vulnerabilities."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -10,3 +10,4 @@ def handle_request(request):
     return process(request)
+def log_user_action(action):
+    logger.info(f"User performed: {action}")
"""

    comments = agent.analyze_security_issues(diff)

    assert len(comments) > 0
    assert any('log injection' in c.comment.lower() for c in comments)
    assert any(c.severity == 'warning' for c in comments)


def test_analyze_code_quality_debug_logging():
    """Test detection of debug logging statements."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/debug.js b/debug.js
index 1234567..abcdefg 100644
--- a/debug.js
+++ b/debug.js
@@ -5,3 +5,4 @@ function processData(data) {
     return result;
 }
+console.log("DEBUG: Processing data");
"""

    comments = agent.analyze_code_quality(diff)

    assert len(comments) > 0
    assert any('debug logging' in c.comment.lower() for c in comments)
    assert any(c.severity == 'info' for c in comments)


def test_analyze_code_quality_todo_comments():
    """Test detection of TODO/FIXME comments."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/feature.py b/feature.py
index 1234567..abcdefg 100644
--- a/feature.py
+++ b/feature.py
@@ -10,3 +10,5 @@ def new_feature():
     pass
+    # TODO: Implement this feature properly
+    # FIXME: This is a temporary workaround
"""

    comments = agent.analyze_code_quality(diff)

    assert len(comments) >= 2
    assert any('todo' in c.comment.lower() or 'fixme' in c.comment.lower() for c in comments)
    assert all(c.severity == 'info' for c in comments if 'todo' in c.comment.lower())


def test_review_comment_dataclass():
    """Test ReviewComment dataclass."""
    comment = ReviewComment(
        file_path='src/test.py',
        line_number=42,
        comment='This is a test comment',
        severity='warning'
    )

    assert comment.file_path == 'src/test.py'
    assert comment.line_number == 42
    assert comment.comment == 'This is a test comment'
    assert comment.severity == 'warning'


def test_analyze_no_issues():
    """Test that clean code produces no issues."""
    agent = GitHubCodeReviewAgent('test', 'test')

    diff = """diff --git a/clean.py b/clean.py
index 1234567..abcdefg 100644
--- a/clean.py
+++ b/clean.py
@@ -1,3 +1,7 @@
 def calculate_total(items):
+    \"\"\"Calculate the total price of items.\"\"\"
     total = 0
     for item in items:
         total += item.price
     return total
"""

    security_comments = agent.analyze_security_issues(diff)
    quality_comments = agent.analyze_code_quality(diff)

    # Clean code should produce no or very few issues
    assert len(security_comments) == 0
    # May have quality suggestions but no errors
    assert all(c.severity != 'error' for c in quality_comments)
