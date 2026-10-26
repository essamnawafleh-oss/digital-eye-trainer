"""Turn the last lines of a log into one GitHub error annotation (readable through the API)."""
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines()
tail = "\n".join(line for line in text[-int(sys.argv[2] if len(sys.argv) > 2 else 120):])
tail = tail.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print(f"::error title={Path(sys.argv[1]).name}::{tail[-60000:]}")
