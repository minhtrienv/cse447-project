#!/usr/bin/env python3
"""Download training data from Project Gutenberg for character-level prediction."""
import os
import ssl
import urllib.request
import re

SOURCES = [
    ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice"),
    ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland"),
    ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "Sherlock Holmes"),
]

def strip_gutenberg_header_footer(text):
    """Remove Project Gutenberg boilerplate."""
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG",
                    "End of the Project Gutenberg", "End of Project Gutenberg"]

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline_after = text.find('\n', idx)
            if newline_after != -1:
                text = text[newline_after + 1:]
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()

def clean_text(text):
    """Clean downloaded text for training."""
    text = strip_gutenberg_header_footer(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove lines that are all caps (chapter headings, etc.) or very short decorative lines
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append('')
            continue
        if len(stripped) < 3:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

def download_text(url, name):
    """Download text from URL."""
    print(f"  Downloading {name}...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            text = resp.read().decode('utf-8', errors='replace')
        return clean_text(text)
    except Exception as e:
        print(f"  Failed to download {name}: {e}")
        return None

def main():
    os.makedirs('data', exist_ok=True)
    corpus_path = 'data/corpus.txt'

    all_text = []
    total_chars = 0
    for url, name in SOURCES:
        text = download_text(url, name)
        if text:
            all_text.append(text)
            total_chars += len(text)
            print(f"  Got {len(text):,} characters from {name}")

    if not all_text:
        print("ERROR: Could not download any training data!")
        return

    combined = '\n\n'.join(all_text)
    with open(corpus_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\nSaved corpus to {corpus_path}")
    print(f"Total: {total_chars:,} characters, {combined.count(chr(10)):,} lines")

if __name__ == '__main__':
    main()
