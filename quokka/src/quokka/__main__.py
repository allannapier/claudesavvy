"""Allow `python -m quokka` invocation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
