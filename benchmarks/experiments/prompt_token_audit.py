"""Optional public-input token tracing; keep disabled in performance runs."""

import hashlib
import json
import struct
from pathlib import Path


def append(path, engine, tokens):
    tokens = list(map(int, tokens))
    digest = hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()
    record = {
        "engine": engine,
        "prompt_tokens": len(tokens),
        "token_ids_sha256_le_u32": digest,
        "token_ids": tokens,
        "scope": "audit_only_not_performance",
    }
    with Path(path).open("a") as output:
        output.write(json.dumps(record) + "\n")
