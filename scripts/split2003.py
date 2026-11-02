# 2003 validation / test split: block = (DOY - 1) // 10; even blocks = test (all paper statistics), odd blocks = validation
# (checkpoint selection). Env SPLIT2003 = all | val | test.
import os, numpy as np

BLOCK_DAYS = 10

def split_part(timestamps):
    """array of 'val' / 'test' / 'other' for each timestamp (any str/datetime-like whose first 10 chars are YYYY-MM-DD)."""
    d = np.array([str(t)[:10] for t in np.asarray(timestamps).ravel()], dtype='datetime64[D]')
    is03 = d.astype('datetime64[Y]') == np.datetime64('2003', 'Y')
    doy = (d - np.datetime64('2003-01-01', 'D')).astype(int) + 1
    blk = (doy - 1) // BLOCK_DAYS
    return np.where(is03, np.where(blk % 2 == 0, 'test', 'val'), 'other')

def split_mask(timestamps, part=None):
    part = (os.environ.get('SPLIT2003', '') if part is None else part).strip().lower()
    if part in ('', 'all'): return np.ones(len(timestamps), bool)
    assert part in ('val', 'test'), f'SPLIT2003 must be val|test|all, got {part!r}'
    return split_part(timestamps) == part

def tag(part=None):
    part = (os.environ.get('SPLIT2003', '') if part is None else part).strip().lower()
    return '' if part in ('', 'all') else f'_split2003{part}'

if __name__ == '__main__':  # self-check
    t = ['2003-01-01 00:00', '2003-01-10 23:59', '2003-01-11 00:00', '2003-10-28 06:00', '2003-11-01 00:00', '2003-11-20 16:34', '2003-12-31 23:50', '2000-04-06 16:47']
    print(dict(zip(t, split_part(t))))
    days = np.arange(np.datetime64('2003-01-01'), np.datetime64('2004-01-01')); p = split_part(days.astype(str))
    print('days: test', (p == 'test').sum(), 'val', (p == 'val').sum())
