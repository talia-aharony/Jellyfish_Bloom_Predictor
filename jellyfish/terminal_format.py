"""Small helpers for consistent terminal section dividers."""


def rule(text: str = "", fill: str = "=", min_len: int = 0) -> str:
    """Return a divider line sized to the supplied text."""
    width = max(len(str(text)), int(min_len), 1)
    return fill * width


def section(title: str, fill: str = "=", min_len: int = 0) -> None:
    """Print a title surrounded by divider lines."""
    line = rule(title, fill=fill, min_len=min_len)
    print(line)
    print(title)
    print(line)


def banner(name: str) -> str:
    """Return a compact banner for model names."""
    return f"── {name} ──"
