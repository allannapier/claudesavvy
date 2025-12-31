"""Setup file for claudesavvy."""

from setuptools import setup, find_packages

setup(
    name="claudesavvy",
    version="2.0.1",
    description="Web-based usage monitoring tool for Claude Code",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "claudesavvy.web": [
            "templates/**/*.html",
            "static/**/*",
        ],
    },
    include_package_data=True,
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "python-dateutil>=2.8.0",
        "Flask>=3.0.0",
        "Jinja2>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "claudesavvy=claudesavvy.cli:main",
        ],
    },
    python_requires=">=3.9",
)
