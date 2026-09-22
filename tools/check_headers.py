"""The check that stands in for running R.

Three things can be wrong after a header rewrite, and each one would only show
up when the script is run:

  1. project_root is USED before it is defined.
  2. an rm() between the definition and the use erases it.
  3. install_load_pkg() is CALLED before the file that defines it is sourced.

All three are answerable by reading line numbers, so they are read here.
"""
import io, glob, re, sys

ROOT = 'D:/usuario_armazenamento/cassio/R/deep_learning_caret/'
files = sorted(glob.glob(ROOT + 'examples/**/*.R', recursive=True)) + \
        sorted(glob.glob(ROOT + 'tests/*.R')) + sorted(glob.glob(ROOT + 'R/*.R'))

DEF = re.compile(r'^\s*(project_root|\.dlc_root|root)\s*<-')
USE = re.compile(r'\bproject_root\b')
RM_ALL = re.compile(r'^\s*rm\(list\s*=\s*ls\(\)\)')
RM_KEEP = re.compile(r'^\s*rm\(list\s*=\s*setdiff\(')
SRC_INST = re.compile(r'source\(file\.path\(project_root,\s*"utils",\s*"install_load_pkg\.R"\)\)')
CALL_INST = re.compile(r'^\s*install_load_pkg\(')

bad = 0
checked = 0
for f in files:
    lines = io.open(f, encoding='utf-8').read().split('\n')
    rel = f[len(ROOT):].replace('\\', '/')
    code = [('' if l.lstrip().startswith('#') else l) for l in lines]

    d_line = next((i for i, l in enumerate(code, 1) if DEF.match(l)), None)
    u_lines = [i for i, l in enumerate(code, 1)
               if USE.search(l) and not DEF.match(l)]
    if d_line is None and not u_lines:
        continue
    checked += 1

    # 1. defined before used
    if u_lines and (d_line is None or d_line > u_lines[0]):
        # a default argument named project_root inside a function is fine
        if 'project_root = ' not in code[u_lines[0] - 1]:
            print('%-52s USE at line %d, definition at %s' % (rel, u_lines[0], d_line))
            bad += 1
            continue

    # 2. an rm() between the definition and the last use must spare it
    if d_line:
        for i, l in enumerate(code, 1):
            if RM_ALL.match(l) and d_line < i < (u_lines[-1] if u_lines else 10**9):
                print('%-52s rm(list = ls()) at line %d wipes the root defined at %d'
                      % (rel, i, d_line))
                bad += 1
                break

    # 3. the installer is sourced before it is called
    src = next((i for i, l in enumerate(code, 1) if SRC_INST.search(l)), None)
    call = next((i for i, l in enumerate(code, 1) if CALL_INST.match(l)), None)
    if call and (src is None or src > call):
        print('%-52s install_load_pkg() called at %d, sourced at %s' % (rel, call, src))
        bad += 1

print('\n%d file(s) checked, %d problem(s)' % (checked, bad))
sys.exit(1 if bad else 0)
