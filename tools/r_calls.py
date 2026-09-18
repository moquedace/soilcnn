"""Does every project function an R script calls actually exist?

Nobody working on this repo through an assistant can run R -- Cassio runs every
script himself -- so the first-run failure most likely to cost him a morning is
a call to a function that does not exist, or a named argument the function does
not take. Both are decidable from the source without parsing R.

Scope: functions defined as `name <- function(...)` in R/*.R. Package calls
(pkg::fn) and base R are out of scope, since their signatures are not here.

Comments and string CONTENTS are removed first, via tools/r_skeleton.py. Doing
this by splitting on '#' is wrong twice over: it cuts at a '#' inside a string
literal, and it leaves string text behind -- so a stop() message reading
"run 04_final_model.R (it calls freeze_selection())" gets reported as a call to
freeze_selection(). Every false positive the first version produced came from
exactly that.

Usage
  python tools/r_calls.py                      -- the scripts listed in MAIN
  python tools/r_calls.py <file.R> [...]       -- specific files
"""
import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE).replace(os.sep, '/') + '/'

_spec = importlib.util.spec_from_file_location(
    'r_skeleton', os.path.join(HERE, 'r_skeleton.py'))
_rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rs)
skeleton = _rs.skeleton

DEF = re.compile(r'^([A-Za-z_.][A-Za-z0-9_.]*)\s*<-\s*function\s*\(', re.M)
CALL = re.compile(r'(?<![A-Za-z0-9_.:$@])([A-Za-z_.][A-Za-z0-9_.]*)\s*\(')
NAMED = re.compile(r'(?<![=<>!])\b([A-Za-z_.][A-Za-z0-9_.]*)\s*=(?!=)')

# do.call(fn, args) invokes fn while naming it as a SYMBOL, so the call regex
# above -- which needs a '(' after the name -- does not see it. That is not a
# corner case here: _b6_resume_check.R runs every one of its training calls
# through do.call(dsm_train, ...), and this checker reported the script as
# never calling it. A typo in that symbol is a runtime error like any other.
#
# The arguments come from a list built elsewhere, so their names cannot be
# checked statically -- only that the target exists. Said out loud in the
# output, because an unchecked argument list that LOOKS checked is worse than
# one that does not.
DOCALL = re.compile(r'\bdo\.call\s*\(\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*,')

KEYWORDS = {'function', 'if', 'for', 'while', 'return', 'switch', 'repeat'}


def _matching(src, i):
    """Index of the ')' closing the '(' at src[i]."""
    depth, j = 0, i
    while j < len(src):
        if src[j] == '(':
            depth += 1
        elif src[j] == ')':
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return len(src) - 1


def _formals(head):
    args, depth, cur = [], 0, ''
    for ch in head:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            args.append(cur)
            cur = ''
        else:
            cur += ch
    args.append(cur)
    return [a.split('=')[0].strip() for a in args if a.split('=')[0].strip()]


def project_functions():
    out = {}
    for f in sorted(glob.glob(ROOT + 'R/*.R')):
        src = skeleton(open(f, encoding='utf-8').read())
        for m in DEF.finditer(src):
            i = src.index('(', m.end() - 1)
            out[m.group(1)] = (_formals(src[i + 1:_matching(src, i)]),
                               os.path.basename(f))
    return out


def check(rel, defined):
    src = open(ROOT + rel, encoding='utf-8').read()
    code = skeleton(src)
    local = set(DEF.findall(code))

    print('\n== %s' % rel)
    ok, unknown, problems = set(), {}, 0

    for m in CALL.finditer(code):
        fn = m.group(1)
        if fn in local or fn in KEYWORDS:
            continue
        if fn not in defined:
            unknown[fn] = unknown.get(fn, 0) + 1
            continue
        ok.add(fn)

        formals = defined[fn][0]
        if '...' in formals:
            continue
        i = m.end() - 1
        inner = code[i + 1:_matching(code, i)]
        top = inner.split('(')[0]          # crude: names before any nested call
        for nm in sorted(set(NAMED.findall(inner))):
            if nm not in formals and nm in top:
                print('  ARG?  %s(%s = ...)' % (fn, nm))
                print('        formals: %s   [%s]'
                      % (', '.join(formals), defined[fn][1]))
                problems += 1

    indirect, missing_target = set(), set()
    for m in DOCALL.finditer(code):
        fn = m.group(1)
        if fn in local or fn in KEYWORDS:
            continue
        if fn in defined:
            indirect.add(fn)
        elif '.' not in fn:          # base R targets such as rbind are fine
            missing_target.add(fn)

    print('  project functions called, all defined: %d' % len(ok))
    if ok:
        print('    %s' % ', '.join(sorted(ok)))
    if indirect:
        print('  via do.call (target exists; ARGUMENT NAMES NOT CHECKED): %s'
              % ', '.join(sorted(indirect)))
    for fn in sorted(missing_target):
        print('  DO.CALL TARGET NOT FOUND UNDER R/: %s' % fn)
        problems += 1
    return problems


MAIN = [
    'examples/soc_stock_0_5cm/_b1_knndm_folds.R',
    'examples/soc_stock_0_5cm/_b3_augmentation.R',
    'examples/soc_stock_0_5cm/_b6_resume_check.R',
]

if __name__ == '__main__':
    defined = project_functions()
    print('project functions defined under R/: %d' % len(defined))
    targets = sys.argv[1:] or MAIN
    total = sum(check(t.replace(ROOT, ''), defined) for t in targets)
    print('\n%d suspicious named argument(s).' % total)
    sys.exit(1 if total else 0)
