import os
import re

frontend_dir = "frontend"
for root, dirs, files in os.walk(frontend_dir):
    for f in files:
        path = os.path.join(root, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            print(f"=== {path} ===")
            matches = re.findall(r'(?i)(key|secret|token|auth|api_key|apikey)', content)
            lines = content.splitlines()
            for idx, line in enumerate(lines, 1):
                if any(k in line.lower() for k in ["key", "secret", "token", "DESMOS_API_KEY"]):
                    print(f"  Line {idx}: {line.strip()[:120]}")
