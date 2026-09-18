"""Prove that an edit touched only comments and string CONTENTS, never code.

Nobody in this session can run R, so a translation sweep over 10 R files is a
sweep nobody can parse-check until the user wakes up. This gives the guarantee
mechanically instead.

It reduces a file to its CODE SKELETON: comments removed, and every string
literal's contents replaced by a placeholder while keeping the quote character
and the string's position. Two files with the same skeleton parse the same way
and execute the same way, whatever their comments and message text say.

Usage
  python r_skeleton.py hash  <file> [...]      -- print each file's skeleton hash
  python r_skeleton.py check <ref.json> <file> [...]  -- compare against saved
  python r_skeleton.py save  <ref.json> <file> [...]  -- save the current hashes
  python r_skeleton.py diff  <fileA> <fileB>   -- show where skeletons differ
"""
import hashlib
import json
import sys


def skeleton(src):
    """Code with comments dropped and string contents blanked.

    R strings are "..." or '...' with backslash escapes; backticks quote names
    (which ARE code -- a renamed backtick name changes behaviour, so their
    contents are kept). A '#' inside any of the three is not a comment.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]

        if c == '#':                                  # comment to end of line
            while i < n and src[i] != '\n':
                i += 1
            continue

        if c in '"\'':                                # string: blank the middle
            quote = c
            i += 1
            while i < n and src[i] != quote:
                if src[i] == '\\':
                    i += 1
                i += 1
            i += 1                                    # past the closing quote
            out.append(quote + '@' + quote)
            continue

        if c == '`':                                  # backtick name: keep it
            out.append(c)
            i += 1
            while i < n and src[i] != '`':
                out.append(src[i])
                i += 1
            if i < n:
                out.append('`')
                i += 1
            continue

        out.append(c)
        i += 1

    # collapse whitespace: re-wrapping a comment must not register as a change,
    # and neither must a trailing space left behind where a comment was
    lines = [' '.join(ln.split()) for ln in ''.join(out).split('\n')]
    return '\n'.join(ln for ln in lines if ln)


def digest(path):
    with open(path, encoding='utf-8') as fh:
        return hashlib.sha256(skeleton(fh.read()).encode()).hexdigest()[:16]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    mode, rest = argv[1], argv[2:]

    if mode == 'hash':
        for p in rest:
            print('%s  %s' % (digest(p), p))
        return 0

    if mode == 'save':
        ref, files = rest[0], rest[1:]
        with open(ref, 'w', encoding='utf-8') as fh:
            json.dump({p: digest(p) for p in files}, fh, indent=1)
        print('saved %d skeleton hash(es) -> %s' % (len(files), ref))
        return 0

    if mode == 'check':
        ref, files = rest[0], rest[1:]
        with open(ref, encoding='utf-8') as fh:
            saved = json.load(fh)
        bad = 0
        for p in files:
            if p not in saved:
                print('NO REFERENCE  %s' % p)
                bad += 1
                continue
            now = digest(p)
            if now == saved[p]:
                print('code unchanged  %s' % p)
            else:
                print('CODE CHANGED    %s  (%s -> %s)' % (p, saved[p], now))
                bad += 1
        print('\n%d of %d file(s) changed code.' % (bad, len(files)))
        return 1 if bad else 0

    if mode == 'diff':
        a, b = rest[0], rest[1:][0]
        sa = skeleton(open(a, encoding='utf-8').read()).split('\n')
        sb = skeleton(open(b, encoding='utf-8').read()).split('\n')
        import difflib
        for ln in difflib.unified_diff(sa, sb, a, b, lineterm='', n=1):
            print(ln)
        return 0

    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
