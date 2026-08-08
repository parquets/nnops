#!/usr/bin/env python3
"""
Add braces to single-statement if/for/while/else bodies in C++ files.

Rules:
  1. Always use {} even for single-statement bodies
  2. { on same line as condition, } on its own line
"""

import re
import os
import sys


def get_indent(line: str) -> int:
    """Return indentation level (number of leading spaces)."""
    return len(line) - len(line.lstrip(' '))


def is_blank_or_comment(line: str) -> bool:
    """Check if line is blank or a pure comment line."""
    stripped = line.strip()
    return not stripped or stripped.startswith('//') or stripped.startswith('/*')


def semicolon_outside_parens(text: str) -> int:
    """Find the first ; that is not inside parentheses. Returns index or -1."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ';' and depth == 0:
            return i
    return -1


def find_matching_paren(text: str, start: int) -> int:
    """Find matching closing paren starting from 'start' (position of '(')."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def is_control_line(line: str):
    """
    Check if a line is an unbraced if/for/while/else.
    Returns (indent, keyword, paren_end) or None.
    keyword is one of: 'if', 'for', 'while', 'else'
    paren_end is the position of the closing ) or -1 for 'else'
    """
    stripped = line.strip()

    # Skip preprocessor lines
    if stripped.startswith('#'):
        return None

    # Skip macro continuations (trailing backslash)
    if stripped.endswith('\\'):
        return None

    # Skip lines that end with { (already braced)
    # Also check after stripping trailing \ (macro continuation)
    check_brace = stripped.rstrip('\\')
    if check_brace.endswith('{'):
        return None

    # Match: if/for/while/else at start of statement
    m = re.match(r'^(\s*)(else\s+)?(if|for|while|else)\b', line)
    if not m:
        return None

    indent = m.group(1)
    keyword = m.group(3)

    # Find the opening paren for if/for/while
    keyword_pos = m.start(3)
    if keyword == 'else':
        # else might be followed by ' if (...)'
        rest = line[m.end(3):].strip()
        if rest.startswith('if'):
            keyword = 'else if'
            paren_pos = line.find('(', m.end(3))
            if paren_pos < 0:
                return None
            paren_end = find_matching_paren(line, paren_pos)
            return (indent, keyword, paren_end)
        else:
            return (indent, keyword, -1)

    # Find ( after the keyword
    paren_pos = line.find('(', keyword_pos)
    if paren_pos < 0:
        return None

    paren_end = find_matching_paren(line, paren_pos)

    # If the closing ) is not on this line, it's a multi-line condition — skip
    if line[paren_end] != ')':
        return None

    return (indent, keyword, paren_end)


def process_file(filepath: str) -> bool:
    """Process a single file. Returns True if changes were made."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    # Detect and preserve original line endings
    if b'\r\n' in raw:
        newline = '\r\n'
    elif b'\n' in raw:
        newline = '\n'
    else:
        newline = '\n'
    text = raw.decode('utf-8', errors='replace')
    lines = text.splitlines(keepends=True)

    original = lines.copy()
    changed = True
    max_iters = 30

    while changed and max_iters > 0:
        max_iters -= 1
        changed = False
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            info = is_control_line(line)

            if info is None:
                new_lines.append(line)
                i += 1
                continue

            indent_str, keyword, paren_end = info
            c_indent = get_indent(line)

            if keyword == 'else':
                # For 'else', body follows on next line
                pass  # fall through to next-line handling
            else:
                # Check if body is on the SAME line after the closing )
                after_paren = line[paren_end + 1:].strip()

                if after_paren and not after_paren.startswith('//') and not after_paren.startswith('{'):
                    # Same-line body (e.g., "if (x) return y;")
                    # Skip if already braced on same line (e.g., "if (x) { stmt; }")
                    body = after_paren
                    comment = ''

                    # Check for trailing comment
                    cmt_pos = body.find('//')
                    if cmt_pos >= 0:
                        comment = body[cmt_pos:].strip()
                        body = body[:cmt_pos].strip()

                    if body:
                        # Extract condition text and build
                        cond_text = line[:paren_end + 1].rstrip()
                        cond_indent = line[:c_indent]
                        prefix = cond_text[len(cond_indent):]  # e.g., "if (x)"

                        if comment:
                            new_lines.append(f"{cond_indent}{prefix} {{\n")
                            new_lines.append(f"{indent_str}    {body} {comment}\n")
                        else:
                            new_lines.append(f"{cond_indent}{prefix} {{\n")
                            new_lines.append(f"{indent_str}    {body}\n")
                        new_lines.append(f"{cond_indent}}}\n")
                        i += 1
                        changed = True
                        continue

            # Next-line body: find the body start
            j = i + 1
            while j < len(lines) and is_blank_or_comment(lines[j]):
                j += 1

            if j < len(lines):
                body_line = lines[j]
                body_stripped = body_line.strip()
                b_indent = get_indent(body_line)

                # Skip macro continuations (body line has trailing \)
                if body_stripped.endswith('\\'):
                    new_lines.append(line)
                    i += 1
                    continue

                if b_indent > c_indent and not body_stripped.startswith('{'):
                    # Single-statement body on next line — add braces
                    body_end = j  # default: single line

                    # Find where body ends (next ; at same indent)
                    for k in range(j, len(lines)):
                        if is_blank_or_comment(lines[k]):
                            continue
                        cur_ind = get_indent(lines[k])
                        # If we exit the indent level, body ended before this
                        if cur_ind <= c_indent and k > j:
                            stripped_check = lines[k].strip()
                            # Check: could be 'else' belonging to our if
                            if stripped_check.startswith('else'):
                                body_end = k - 1
                                break
                            body_end = k - 1
                            break
                        if cur_ind == b_indent and semicolon_outside_parens(lines[k]) >= 0:
                            body_end = k
                            break
                        # Could be continuation line (more indent)
                        body_end = k

                    # Extract body lines
                    body_lines = lines[j:body_end + 1]

                    # Build output
                    cond_line = line.rstrip()
                    if keyword == 'else':
                        new_lines.append(f"{cond_line} {{\n")
                    else:
                        new_lines.append(f"{cond_line} {{\n")

                    new_lines.extend(body_lines)
                    new_lines.append(f"{line[:c_indent]}}}\n")

                    i = body_end + 1
                    changed = True
                    continue

            # No match — keep as-is
            new_lines.append(line)
            i += 1

        lines = new_lines

    if lines != original:
        # Ensure all lines have proper endings
        fixed = []
        for l in lines:
            if not l.endswith(('\n', '\r\n')):
                fixed.append(l + newline)
            else:
                fixed.append(l)
        with open(filepath, 'wb') as f:
            f.write(''.join(fixed).encode('utf-8'))
        return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_braces.py <root_dir> [--dry-run]")
        print("  root_dir: root directory to scan")
        print("  --dry-run: print files that would be processed, don't modify")
        sys.exit(1)

    root = sys.argv[1]
    dry_run = '--dry-run' in sys.argv

    extensions = {'.hpp', '.cpp', '.h'}
    skip_dirs = {'build', '.git', '__pycache__', 'node_modules'}
    modified = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if os.path.splitext(fname)[1] not in extensions:
                continue
            fpath = os.path.join(dirpath, fname)
            if dry_run:
                print(f"  Would process: {fpath}")
            else:
                if process_file(fpath):
                    modified.append(fpath)
                    print(f"  Modified: {fpath}")

    if dry_run:
        print("\nDry run complete. Add --apply instead of --dry-run to modify files.")
    else:
        print(f"\nModified {len(modified)} file(s).")


if __name__ == '__main__':
    main()
