import re
import os

EMOJI_RANGES = [
    (0x1F600, 0x1F64F),
    (0x1F300, 0x1F5FF),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0xFE0F, 0xFE0F),
    (0x200D, 0x200D),
    (0x20E3, 0x20E3),
]

pattern_parts = []
for start, end in EMOJI_RANGES:
    if start == end:
        pattern_parts.append(chr(start))
    else:
        pattern_parts.append(chr(start) + "-" + chr(end))

emoji_pattern = re.compile("[" + "".join(pattern_parts) + "]+", flags=re.UNICODE)

project = r"C:\Users\manas\clg\ehat\charging-station-load-prediction"
extensions = [".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini"]
changes = []

for root, dirs, files in os.walk(project):
    for f in files:
        path = os.path.join(root, f)
        ext = os.path.splitext(f)[1]
        if ext in extensions:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            if emoji_pattern.search(content):
                new_content = emoji_pattern.sub("", content)
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(new_content)
                    changes.append(path)
                    matches = emoji_pattern.findall(content)
                    print(f"{path}: removed {len(matches)} emoji(s)")

if not changes:
    print("No emojis found in any file.")
else:
    print(f"\nFixed {len(changes)} file(s).")
