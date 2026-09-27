"""Where dsm_prepare()'s extraction time goes: a model of its reading plan.

Written on 2026-09-27, after P1 of the single-pass extraction (68.2 min for
181 bands at 15 workers), to decide what to optimise next. It reads only
metadata -- the TIFF headers and the point tables -- never the pixels, and
runs in about a minute.

1. The points. The dev subsample is rebuilt from the GPKG: whole 2-degree
   blocks, the blocks of the rows of the point table. Blocks whose every
   point the QC dropped cannot be recovered that way, so the rebuild is a
   close approximation (4,124 of 4,154 points, 106 of 108 blocks).
2. The reading plan of R/prepare.R, re-implemented; it reproduces the
   plan the P1 log printed (169 reads vs 173, 42 chunks, 0.206% vs 0.209%
   of the cells, largest read 12 MB).
3. Per band, the COMPRESSED bytes of the strips (rows) the plan touches,
   from each TIFF's StripByteCounts: what GDAL has to read from the disk
   and decompress. The rasters are one-row strips, so a touched row is
   read and decompressed whole.
4. Two models of a batch of 15 bands, fitted to the 13 batch times the P1
   log printed: disk-bound (time ~ the SUM of the batch's bytes, one disk
   shared) or CPU-bound (time ~ the MAX, the slowest core). On 2026-09-27
   the disk model fitted with R^2 0.97 and the CPU model with 0.24.

The batch times below are that run's; to model another run, replace them.

Run: python tools/extraction_io_model.py
"""
import csv, io, math, os, sqlite3, struct, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')
META = ROOT + '/outputs/metadata/soc_stock_modeling/soc_stock_0_5cm'
PTS  = ROOT + '/data/processed/soc_stock_modeling/soc_stock_0_5cm/full_modeling_dataset_raw.csv'
GPKG = ROOT + '/data/raw/wosis_profile_soc_stock_spline_clean_preSpline.gpkg'
XMIN, YMAX, RES = -179.99999813907394, 83.63297651181205, 0.002245798111732948
NROW, NCOL = 63721, 160298
H, CHUNK, GAP, MAXC = 7, 1000, 256, 4096

def read_csv2(path):
    with io.open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter=';'))
def num(s): return float(s.replace(',', '.'))

# ── 1. the points ────────────────────────────────────────────────────────────
def gpkg_xy(blob):
    flags = blob[3]
    if flags & 16: return None                      # empty geometry
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    wkb = blob[8 + env:]
    bo = '<' if wkb[0] == 1 else '>'
    return struct.unpack(bo + 'dd', wkb[5:21])

con = sqlite3.connect('file:%s?mode=ro' % GPKG, uri=True)
g = [(str(pid), gpkg_xy(b)) for pid, b in con.execute(
    'select profile_id, geom from "wosis_profile_soc_stock_spline_clean_preSpline"')]
g = [(p, xy) for p, xy in g if xy is not None]
gx = np.array([xy[0] for _, xy in g]); gy = np.array([xy[1] for _, xy in g])
gpid = [p for p, _ in g]
print('GPKG points: %d' % len(g))

pt = read_csv2(PTS)
old_pid = set(r['profile_id'] for r in pt)
bx = np.floor((gx - gx.min()) / 2).astype(int); by = np.floor((gy - gy.min()) / 2).astype(int)
blk = list(zip(bx, by))
dev_blocks = set(b for b, p in zip(blk, gpid) if p in old_pid)
dev_idx = [i for i, b in enumerate(blk) if b in dev_blocks]
print('dev blocks rebuilt: %d (log: 108) | dev points: %d (log: 4,154)' % (len(dev_blocks), len(dev_idx)))
old_x = np.array([num(r['x']) for r in pt]); old_y = np.array([num(r['y']) for r in pt])

def rowcol(x, y):
    col = np.floor((x - XMIN) / RES).astype(int) + 1
    row = np.floor((YMAX - y) / RES).astype(int) + 1
    return row, col

sets = {'old (3,766, the point table)': rowcol(old_x, old_y),
        'new (4,154, every point)':     rowcol(gx[dev_idx], gy[dev_idx]),
        'full (41,385)':                rowcol(gx, gy)}

# ── 2. the reading plan ──────────────────────────────────────────────────────
def column_groups(cs, h, gap, max_cols):
    groups, start = [], 0
    g_lo, g_hi = cs[0] - h, cs[0] + h
    for k in range(1, len(cs)):
        lo, hi = cs[k] - h, cs[k] + h
        if lo - g_hi <= gap and (max(g_hi, hi) - g_lo + 1) <= max_cols:
            g_hi = max(g_hi, hi)
        else:
            groups.append((start, k)); start = k; g_lo, g_hi = lo, hi
    groups.append((start, len(cs)))
    return groups

def plan(rows, cols, row_aware=False):
    """Reads as (r0, r1, c0, c1). row_aware: split a column group where its
    points' windows leave a run of rows no point needs."""
    ok = (rows - H >= 1) & (rows + H <= NROW) & (cols - H >= 1) & (cols + H <= NCOL)
    reads, chunks = [], 0
    for cs in range(1, NROW + 1, CHUNK):
        ce = min(cs + CHUNK - 1, NROW)
        idx = np.where(ok & (rows >= cs) & (rows <= ce))[0]
        if idx.size == 0: continue
        chunks += 1
        idx = idx[np.argsort(cols[idx], kind='stable')]
        for a, b in column_groups(list(cols[idx]), H, GAP, MAXC):
            gi = idx[a:b]
            if not row_aware:
                reads.append((rows[gi].min() - H, rows[gi].max() + H, cols[gi].min() - H, cols[gi].max() + H))
                continue
            gi = gi[np.argsort(rows[gi], kind='stable')]
            s = 0
            for k in range(1, len(gi) + 1):
                if k == len(gi) or rows[gi[k]] - H > rows[gi[k - 1]] + H + 1:
                    sub = gi[s:k]
                    reads.append((rows[sub].min() - H, rows[sub].max() + H, cols[sub].min() - H, cols[sub].max() + H))
                    s = k
    return reads, chunks, int((~ok).sum())

def touched(reads):
    m = np.zeros(NROW + 1, dtype=bool)
    for r0, r1, _, _ in reads: m[r0:r1 + 1] = True
    return m[1:]

# ── 3. the compressed bytes of every strip, per band ─────────────────────────
def strip_byte_counts(path):
    with open(path, 'rb') as f:
        head = f.read(16)
        bo = '<' if head[:2] == b'II' else '>'
        big = struct.unpack(bo + 'H', head[2:4])[0] == 43
        ifd = struct.unpack(bo + ('Q' if big else 'I'), head[8:16] if big else head[4:8])[0]
        f.seek(ifd)
        n = struct.unpack(bo + ('Q' if big else 'H'), f.read(8 if big else 2))[0]
        ent, fmt, off = (20, 'HHQQ', 12) if big else (12, 'HHII', 8)
        raw = f.read(n * ent)
        for k in range(n):
            tag, typ, cnt, val = struct.unpack(bo + fmt, raw[k * ent:(k + 1) * ent])
            if tag == 279:
                code = {3: 'H', 4: 'I', 16: 'Q'}[typ]
                nbytes = struct.calcsize(code) * cnt
                if nbytes <= (8 if big else 4):
                    data = raw[k * ent + off:k * ent + off + nbytes]
                else:
                    f.seek(val); data = f.read(nbytes)
                return np.array(struct.unpack(bo + code * cnt, data), dtype=np.int64)
    raise RuntimeError('no StripByteCounts in ' + path)

types = read_csv2(META + '/predictor_type_table.csv')
rtab = {r['predictor']: r['raster_file'] for r in read_csv2(META + '/raster_table_used.csv')}
preds = [r['predictor'] for r in types]
sbc = np.vstack([strip_byte_counts(rtab[p]) for p in preds])        # 181 x 63721
print('bands: %d | compressed total %.0f GB' % (len(preds), sbc.sum() / 1e9))

# ── 4. the measured batches, and the two models ─────────────────────────────
# cumulative minutes at bands 15, 30, ..., 180, 181 (P1 log, 2026-09-27)
cum = [11.2, 17.4, 19.3, 25.8, 28.2, 38.6, 47.8, 55.2, 62.7, 63.4, 64.1, 67.7, 68.2]
meas = np.diff([0.0] + cum) * 60                                     # seconds
batches = [list(range(s, min(s + 15, len(preds)))) for s in range(0, len(preds), 15)]

res = {}
for name, (rows, cols) in sets.items():
    for ra in (False, True):
        reads, chunks, n_edge = plan(rows, cols, row_aware=ra)
        m = touched(reads)
        per_band = (sbc[:, m]).sum(axis=1)                             # bytes
        res[(name, ra)] = (reads, chunks, m, per_band)
        print('%-30s %-10s reads %5d  chunks %2d  rows touched %6d (%.1f%% of rows)  read %.0f GB compressed%s'
              % (name, 'row-aware' if ra else 'current', len(reads), chunks, m.sum(), 100 * m.mean(),
                 per_band.sum() / 1e9, ('  | points too close to the edge: %d' % n_edge) if not ra else ''))

reads, chunks, m, per_band = res[('new (4,154, every point)', False)]
cells = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in reads)
big = max((r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in reads) * 8 / 1e6
print('\ncheck against the P1 log: %d reads / %d chunks / %.3g%% of cells / largest %.0f MB  (log: 173 / 42 / 0.209%% / 12 MB)'
      % (len(reads), chunks, 100 * cells / (NROW * NCOL), big))

sum_b = np.array([per_band[b].sum() for b in batches]) / 1e9
max_b = np.array([per_band[b].max() for b in batches]) / 1e9
def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, pred, r2
(kd, cd), pd_, r2d = fit(sum_b, meas)
(kc, cc), pc_, r2c = fit(max_b, meas)
print('\nbatch  measured  | disk model (sum of bytes) | CPU model (max of bytes)')
for i in range(len(batches)):
    print('  %2d   %6.1f min | %5.1f GB -> %5.1f min      | %5.2f GB -> %5.1f min'
          % (i + 1, meas[i] / 60, sum_b[i], pd_[i] / 60, max_b[i], pc_[i] / 60))
print('R^2: disk %.3f | CPU %.3f' % (r2d, r2c))
print('disk model: %.0f MB/s from the HDD across the 15 workers (intercept %.0f s per batch)' % (1e3 / kd, cd))
print('whole run : %.0f GB compressed in %.1f min = %.0f MB/s on average'
      % (per_band.sum() / 1e9, sum(meas) / 60, per_band.sum() / 1e6 / sum(meas)))

# the old run: same batches, the old plan
_, _, _, pb_old = res[('old (3,766, the point table)', False)]
old_pred = sum(kd * pb_old[b].sum() / 1e9 + cd for b in batches) / 60
new_pred = sum(pd_) / 60
print('\nthe old run by the disk model: %.1f min (measured 60.6) | the new: %.1f (measured 68.2)' % (old_pred, new_pred))
for name in sets:
    for ra in (False, True):
        pb = res[(name, ra)][3]
        print('  predicted extraction  %-30s %-10s %6.1f min'
              % (name, 'row-aware' if ra else 'current', sum(kd * pb[b].sum() / 1e9 + cd for b in batches) / 60))
