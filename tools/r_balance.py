"""Bracket balance of R files, with real line numbers.

A minimal R lexer: "..." and '...' strings with backslash escapes, `...`
names, r"(...)" raw strings, and # comments to the end of the line. Every
(, [ and { must close in order. This is not a parser -- it cannot say a call
is well formed -- but an unbalanced bracket is the most common way an edit
breaks a file that nobody here can source.
"""
import io, re, sys

def check(path):
    src = io.open(path, encoding='utf-8').read()
    i, n, line = 0, len(src), 1
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    while i < n:
        c = src[i]
        if c == '\n':
            line += 1; i += 1; continue
        if c == '#':
            while i < n and src[i] != '\n': i += 1
            continue
        m = re.match(r'[rR](["\'])(-*)([\(\[\{])', src[i:i + 40])
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] in '._')):
            q, dashes, open_ = m.group(1), m.group(2), m.group(3)
            close = {'(': ')', '[': ']', '{': '}'}[open_] + dashes + q
            j = src.find(close, i + len(m.group(0)))
            if j < 0: return 'unterminated raw string from line %d' % line
            line += src.count('\n', i, j); i = j + len(close); continue
        if c in '"\'`':
            start = line; i += 1
            while i < n and src[i] != c:
                if src[i] == '\\' and c != '`': i += 1
                elif src[i] == '\n': line += 1
                i += 1
            if i >= n: return 'unterminated %s string from line %d' % (c, start)
            i += 1; continue
        if c in '([{': stack.append((c, line))
        elif c in ')]}':
            if not stack or stack[-1][0] != pairs[c]:
                return 'unexpected %s at line %d (open: %s)' % (c, line, stack[-1] if stack else None)
            stack.pop()
        i += 1
    if stack: return 'unclosed %s from line %d' % stack[-1]
    return None

bad = 0
for f in sys.argv[1:]:
    r = check(f)
    print('%-56s %s' % (f, r or 'balanced'))
    bad += r is not None
sys.exit(1 if bad else 0)
