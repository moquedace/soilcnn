"""Catch the R mistakes that cost a round trip, without running R.

tests/test_sources_parse.R already parses every file -- but only Cassio can run
it, and parse() reports just the FIRST error per file, so one bad line hides
every other one behind it. This finds them all at once, and adds the project
rules that are not syntax errors at all.

Two checks, each chosen because it has actually happened here. A delimiter
balance check was written and then DELETED: three versions of it still
miscounted against files that parse perfectly, and parse() does that job
correctly the moment the suite runs. A lint that cries wolf on working code
gets ignored on the day it is right, and there is no credit for breadth here.

  else-at-top-level   R closes `x <- if (cond) expr` at the end of the line, so
                      a bare `else` starting the next one is a syntax error --
                      legal only inside an open delimiter. _b1_knndm_folds.R
                      shipped with one and failed the parse test.

  native pipe         this project uses %>% always and |> never. Not a syntax
                      error; a house rule, and one that reads as correct code.

Comments and string contents are blanked first, via tools/r_skeleton.py, so a
brace inside a message() and a '#' inside a string cannot mislead it.

Usage
  python tools/r_lint.py                 -- R/, examples/soc_stock_0_5cm/, tests/
  python tools/r_lint.py <file.R> [...]  -- specific files
Exit status is 1 if anything was found, so it can gate a commit.
"""
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE).replace(os.sep, '/') + '/'

_spec = importlib.util.spec_from_file_location(
    'r_skeleton', os.path.join(HERE, 'r_skeleton.py'))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)

OPEN, CLOSE = '([{', ')]}'


def _blank(src):
    """Comments, string contents and backtick names blanked; lines preserved.

    ONE PASS OVER THE WHOLE FILE, not line by line. R string literals may span
    newlines -- tests/helper.R opens one in `cat(paste0(...), "` and closes it
    on the next line -- so a per-line reduction treats the continuation as code
    and every delimiter after it is counted against the wrong opener.

    Backtick-quoted names are blanked too, which r_skeleton deliberately does
    NOT do: it keeps them because renaming one changes behaviour. For counting
    delimiters they have to be atoms, or `lapply(p$folds, \\`[[\\`, "train")`
    pushes two `[` that never close.

    These two causes accounted for every one of the 58, then 31, false findings
    the first two versions of this lint produced. A check that cries wolf on
    working code gets ignored on the day it is right.
    """
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c == '#':
            while i < n and src[i] != '\n':
                out.append(' ')
                i += 1
        elif c in '"\'`':
            quote = c
            out.append(' ')
            i += 1
            while i < n and src[i] != quote:
                # newlines are kept so line numbers survive a multi-line string
                out.append('\n' if src[i] == '\n' else ' ')
                if src[i] == '\\':
                    i += 1
                    if i < n:
                        out.append(' ')
                i += 1
            out.append(' ')
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out).split('\n')


def check_else(raw, code):
    """A bare `else` is only legal while a delimiter is still open."""
    out, depth = [], 0
    for i, ln in enumerate(code):
        if ln.strip().startswith('else') and depth == 0:
            out.append((i + 1, 'top-level `else`: R already closed the `if` at '
                               'the end of the previous line', raw[i].strip()))
        depth += sum(ln.count(c) for c in OPEN) - sum(ln.count(c) for c in CLOSE)
    return out


def check_native_pipe(raw, code):
    out = []
    for i, ln in enumerate(code):
        # |> but not ||> and not a comparison such as `a | > b` (not valid R
        # anyway); %>% is the house pipe
        if '|>' in ln:
            out.append((i + 1, 'native pipe |> -- this project uses %>%',
                        raw[i].strip()))
    return out


CHECKS = (check_else, check_native_pipe)


def lint(path):
    src = open(path, encoding='utf-8').read()
    raw = src.split('\n')
    code = _blank(src)
    # _blank blanks in ONE pass over the whole file, so this is the invariant
    # that says line numbers still mean something. Without it, passing the
    # wrong shape here silently folded every finding onto line 1 and the whole
    # repo reported clean.
    assert len(code) == len(raw), (
        'line count moved in %s: %d vs %d' % (path, len(code), len(raw)))
    found = []
    for fn in CHECKS:
        found.extend(fn(raw, code))
    return sorted(found)


# A LINT THAT REPORTS NOTHING LOOKS EXACTLY LIKE A LINT THAT CHECKS NOTHING.
#
# This one did, for one commit: lint() was passing the wrong shape to _blank(),
# every finding folded onto line 1, and the whole repo came back "0 findings".
# The fixture below distinguishes the two states, and it is the same distinction
# tests/helper.R now enforces for the R suite.
SELFTEST = '''x <- if (TRUE) 1
     else 2
y <- c(
  a = if (TRUE) 1
      else 2
)
f <- function() {
  if (TRUE) 1
  else 2
}
s <- "a string with # and { and an else
      spanning two lines"
z <- a |> head()
'''
SELFTEST_EXPECT = [(2, 'else'), (13, 'pipe')]


def selftest():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'lint_fixture.R')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(SELFTEST)
    got = [(ln, 'else' if 'else' in why else 'pipe') for ln, why, _ in lint(path)]
    ok = got == SELFTEST_EXPECT
    print('selftest: %s' % ('PASS' if ok else 'FAIL'))
    if not ok:
        print('  expected %s' % SELFTEST_EXPECT)
        print('  got      %s' % got)
        print('  line 2 is top level and must be flagged; lines 5 and 9 sit')
        print('  inside ( and { and must NOT be; line 13 is the native pipe.')
    return 0 if ok else 1


def main(argv):
    if '--selftest' in argv:
        return selftest()
    targets = argv[1:]
    if not targets:
        targets = sorted(glob.glob(ROOT + 'R/*.R')
                         + glob.glob(ROOT + 'examples/soc_stock_0_5cm/*.R')
                         + glob.glob(ROOT + 'tests/*.R'))
    total = 0
    for p in targets:
        found = lint(p)
        if found:
            # relpath raises across Windows drives, and a file passed by hand
            # is often on another one -- that is exactly how this tool gets
            # self-tested against a fixture.
            try:
                label = os.path.relpath(p, ROOT).replace(os.sep, '/')
            except ValueError:
                label = p.replace(os.sep, '/')
            print('\n== %s' % label)
            for ln, why, txt in found:
                print('  %4d  %s' % (ln, why))
                print('        %s' % txt[:96])
            total += len(found)
    print('\n%d finding(s) across %d file(s).' % (total, len(targets)))
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
