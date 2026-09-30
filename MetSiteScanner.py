#!/usr/bin/env python3
"""
MetSiteScanner.py -- find mono- or di-metal binding sites in a peptide PDB.

Generalized over Cu / Zn / Fe / Co / Ni (metal parameters live in one table,
METALS, below -- add a new metal there and everything else works unchanged).
Donor recognition is deliberately kept to what actually exists in the
current system: His (HIE/HID) imidazole N, plus the N-terminal backbone
amine. No Cys/Asp/Glu/Met code, because this peptide has none.

USAGE
    python3 MetSiteScanner.py input.pdb --mono Cu4 [options]
    python3 MetSiteScanner.py input.pdb --di   Cu4 [options]
    python3 MetSiteScanner.py input.pdb --both Cu4 [options]

    "Cu4"  = metal symbol + coordination number (CN), e.g. Cu4, Zn4, Fe6.
             The CN is parsed and checked against that metal's typical
             range BEFORE any search runs (see analyze_request()).

    --mono  find single-metal sites for the given CN
    --di    find metal-metal PAIRS whose separation falls in --sep,
            built from single-metal candidates that meet the CN/geometry
            criteria, optionally sharing one bridging donor
    --both  run --mono then --di with the same METAL+CN, written into ONE
            combined <stem>_<metal><CN>_both.pdb (mono sites + di pairs).
            Still two independent searches — di pairs are NOT guaranteed
            to reuse the same sites mono reports (di optimizes for
            separation/bridging, mono for standalone score)
    --only  require exactly CN donors per site [default]
    --upto  accept sites with 1..CN donors, not just CN (applies to
            --mono and --di, since --di is built from mono candidates)

OPTIONS
    --don-cutoff A     max donor-donor distance to treat as "local" [7.0]
    --clash A          min metal - non-coordinating-atom distance   [1.8]
    --sep MIN MAX      (--di) target metal-metal separation window
                        default = metal-specific value in METALS table
    --no-bridge        (--di) forbid a shared donor between the two metals
    --no-nterm         exclude the N-terminal amine as a donor
    --max-neighbors N  cap on local donor pool per seed (speed knob) [10]
    --top N            number of ranked sites/pairs to keep & write  [5]

OUTPUT
    <stem>_<metal><CN>_mono.pdb   or   <stem>_<metal><CN>_di.pdb
    <stem>_<metal><CN>_both.pdb   (with --both, mono+di combined)
    plus a ranked text summary printed to screen.
"""
import sys, re, math, argparse, textwrap
from itertools import combinations
import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# 1. METAL PARAMETER TABLE  -- the one place to edit for a new metal
# --------------------------------------------------------------------------
# cn_range : (min, max) coordination numbers normally seen for this metal
# r_n/r_o  : ideal metal-donor bond length (A) for N- and O-donors
# sep      : default target window (A) for a --di metal-metal search
# geom     : CN -> target donor-metal-donor angle (deg) used for the
#            angle-consistency score. None = no angle preference (CN=1).
METALS = {
    'CU': dict(cn_range=(1, 4), r_n=2.00, r_o=2.00, sep=(4.0, 7.0),
               geom={1: None, 2: 150.0, 3: 120.0, 4: 109.5}),
    'ZN': dict(cn_range=(2, 4), r_n=2.05, r_o=2.10, sep=(3.0, 6.0),
               geom={2: 150.0, 3: 120.0, 4: 109.5}),
    'FE': dict(cn_range=(2, 6), r_n=2.15, r_o=2.05, sep=(3.0, 6.0),
               geom={2: 150.0, 3: 120.0, 4: 109.5, 5: 100.0, 6: 90.0}),
    'CO': dict(cn_range=(2, 6), r_n=2.10, r_o=2.05, sep=(3.0, 6.0),
               geom={2: 150.0, 3: 120.0, 4: 109.5, 5: 100.0, 6: 90.0}),
    'NI': dict(cn_range=(2, 6), r_n=2.05, r_o=2.05, sep=(3.0, 6.0),
               geom={2: 150.0, 3: 120.0, 4: 109.5, 5: 100.0, 6: 90.0}),
}

# --------------------------------------------------------------------------
# 2. CN token parsing + up-front analysis (done BEFORE any search)
# --------------------------------------------------------------------------
def analyze_request(token, mode):
    m = re.match(r'^([A-Za-z]+)(\d+)$', token.strip())
    if not m:
        sys.exit(f"[ERROR] Can't parse '{token}'. Expected METAL+CN, e.g. Cu4")
    metal, cn = m.group(1).upper(), int(m.group(2))
    if metal not in METALS:
        sys.exit(f"[ERROR] Unknown metal '{metal}'. Known: {', '.join(METALS)}")
    p = METALS[metal]
    lo, hi = p['cn_range']
    print(f"\n  Request : --{mode} {token}")
    print(f"  Metal   : {metal}   CN requested = {cn}")
    print(f"  Typical CN range for {metal}: {lo}-{hi}")
    if not (lo <= cn <= hi):
        print(f"  [!] WARNING: CN {cn} is outside the typical range for {metal} "
              f"-- proceeding anyway (you asked for it explicitly).")
    target_angle = p['geom'].get(cn)
    print(f"  Ideal {metal}-N   : {p['r_n']:.2f} A")
    print(f"  Ideal {metal}-O   : {p['r_o']:.2f} A")
    print(f"  Angle target      : {target_angle if target_angle else 'none (CN=1)'} deg")
    if mode == 'di':
        print(f"  Default M-M sep   : {p['sep'][0]:.1f}-{p['sep'][1]:.1f} A")
    print()
    return metal, cn, p


# --------------------------------------------------------------------------
# 3. PDB parsing
# --------------------------------------------------------------------------
def parse_pdb(path):
    atoms, atom_lines, header = [], [], []
    try:
        lines = open(path).readlines()
    except OSError as e:
        sys.exit(f"[ERROR] {e}")
    for line in lines:
        rec = line[:6].strip()
        if rec in ('ATOM', 'HETATM'):
            try:
                atoms.append({
                    'record': rec, 'atname': line[12:16].strip(),
                    'resname': line[17:20].strip(),
                    'chain': line[21] if len(line) > 21 else ' ',
                    'resnum': int(line[22:26]),
                    'x': float(line[30:38]), 'y': float(line[38:46]), 'z': float(line[46:54]),
                })
            except (ValueError, IndexError):
                pass
            atom_lines.append(line)
        elif rec not in ('TER', 'END', 'ENDMDL'):
            header.append(line)
    if not atoms:
        sys.exit(f"[ERROR] No ATOM/HETATM records in {path}")
    return atoms, atom_lines, header


# --------------------------------------------------------------------------
# 4. Donor extraction -- His imidazole N + (optional) N-terminal amine
# --------------------------------------------------------------------------
def get_donors(atoms, include_nterm=True):
    donors = []
    for a in atoms:
        if a['resname'] == 'HIE' and a['atname'] == 'ND1':
            donors.append({**a, 'elem': 'N', 'role': 'His(HIE)-ND1'})
        elif a['resname'] == 'HID' and a['atname'] == 'NE2':
            donors.append({**a, 'elem': 'N', 'role': 'His(HID)-NE2'})
        elif a['resname'] == 'HIP':
            pass  # doubly-protonated His: no free lone pair, skip (as before)

    if include_nterm:
        first_resnum = {}
        for a in atoms:
            if a['record'] == 'ATOM':
                first_resnum.setdefault(a['chain'], a['resnum'])
                first_resnum[a['chain']] = min(first_resnum[a['chain']], a['resnum'])
        for a in atoms:
            if a['atname'] == 'N' and a['resnum'] == first_resnum.get(a['chain']):
                donors.append({**a, 'elem': 'N', 'role': 'N-term'})
    return donors


# --------------------------------------------------------------------------
# 5. Geometry: multi-start least-squares metal placement + angle scoring
# --------------------------------------------------------------------------
def d3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

def fit_metal_position(pts, ideal):
    """Multi-start Nelder-Mead: avoids landing on an arbitrary point of the
    degenerate sphere/circle of solutions when CN=1 or CN=2."""
    pts = [np.asarray(p) for p in pts]
    centroid = np.mean(pts, axis=0)
    starts = [centroid]
    rng = np.random.default_rng(0)
    for _ in range(6):
        starts.append(centroid + rng.normal(scale=ideal, size=3))
    if len(pts) >= 2:
        n = (pts[1] - pts[0])
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            perp = np.cross(n / norm, [0, 0, 1])
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(n / norm, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            starts.append(centroid + perp * ideal)

    best_x, best_f = None, np.inf
    for x0 in starts:
        r = minimize(lambda p: sum((np.linalg.norm(p - c) - ideal) ** 2 for c in pts),
                     x0, method='Nelder-Mead',
                     options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
        if r.fun < best_f:
            best_f, best_x = r.fun, r.x
    return tuple(float(v) for v in best_x), float(best_f)

def angle_score(metal_xyz, pts, target_deg):
    """Mean squared deviation (deg^2) of donor-metal-donor angles from
    target_deg, scaled down so it's comparable in magnitude to the A^2
    distance-fit residual."""
    if target_deg is None or len(pts) < 2:
        return 0.0
    m = np.asarray(metal_xyz)
    vecs = [np.asarray(p) - m for p in pts]
    devs = []
    for i, j in combinations(range(len(vecs)), 2):
        va, vb = vecs[i], vecs[j]
        cosang = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        devs.append((ang - target_deg) ** 2)
    return (sum(devs) / len(devs)) / 1000.0   # deg^2 -> comparable scale


# --------------------------------------------------------------------------
# 6. Mono-metal site search (spatially pre-filtered, not brute-force global)
# --------------------------------------------------------------------------
def dkey(d):
    return (d['chain'], d['resnum'], d['atname'])

def bonded_hydrogen_keys(donors, atoms, bond_cut=1.3):
    """For each donor atom, find its own directly-bonded H atom(s) (e.g. the
    H1/H2/H3 protons still sitting on an N-terminal NH3+ that is itself the
    donor). These must be exempted from the clash check along with the donor
    atom -- a metal binding that N is expected to sit close to those protons
    (that's the whole point of the bond), so flagging it as a steric clash
    is a false positive, not a real problem with the site."""
    all_xyz = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    out = {}
    for d in donors:
        dxyz = np.array([d['x'], d['y'], d['z']])
        near = np.where(np.linalg.norm(all_xyz - dxyz, axis=1) < bond_cut)[0]
        keys = {dkey(atoms[i]) for i in near
               if atoms[i]['atname'].startswith('H') and (atoms[i]['chain'], atoms[i]['resnum']) == (d['chain'], d['resnum'])}
        out[dkey(d)] = keys
    return out

def find_mono_sites(donors, atoms, cn, params, cfg, dedup=True):
    from scipy.spatial import cKDTree
    coords = np.array([[d['x'], d['y'], d['z']] for d in donors])
    tree = cKDTree(coords)
    all_xyz = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    ideal = params['r_n']            # His-N / N-term-N donors only in this system
    bonded_h = bonded_hydrogen_keys(donors, atoms)

    sizes = range(1, cn + 1) if cfg.upto else [cn]

    seen_combos = set()
    raw = []
    for seed in range(len(donors)):
        nbr_idx = tree.query_ball_point(coords[seed], cfg.don_cutoff)
        if len(nbr_idx) < min(sizes):
            continue
        # keep only the closest max_neighbors around this seed (speed knob)
        nbr_idx = sorted(nbr_idx, key=lambda i: np.linalg.norm(coords[i] - coords[seed]))
        nbr_idx = nbr_idx[:cfg.max_neighbors]
        if seed not in nbr_idx:
            nbr_idx = [seed] + nbr_idx[:cfg.max_neighbors - 1]

        for size in sizes:
            if len(nbr_idx) < size:
                continue
            for combo in combinations(sorted(nbr_idx), size):
                if combo in seen_combos:
                    continue
                seen_combos.add(combo)
                pts = [coords[i] for i in combo]
                if any(np.linalg.norm(pts[a] - pts[b]) > cfg.don_cutoff
                       for a in range(size) for b in range(a + 1, size)):
                    continue

                xyz, dist_res = fit_metal_position(pts, ideal)
                dists = [float(np.linalg.norm(np.asarray(xyz) - p)) for p in pts]
                if max(dists) > ideal + cfg.max_dist:
                    continue

                exempt = {dkey(donors[i]) for i in combo}
                for i in combo:
                    exempt |= bonded_h[dkey(donors[i])]
                clash = any(d3(xyz, tuple(xyz_a)) < cfg.clash
                           for a, xyz_a in zip(atoms, all_xyz)
                           if dkey(a) not in exempt)
                if clash:
                    continue

                ascore = angle_score(xyz, pts, params['geom'].get(size))
                raw.append({
                    'donors': [donors[i] for i in combo],
                    'xyz': xyz, 'dists': dists,
                    'score': dist_res + ascore, 'cn': size,
                })

    raw.sort(key=lambda s: s['score'])
    if not dedup:
        return raw

    used_donor_keys, used_xyz, final = set(), [], []
    for s in raw:
        keys = {dkey(d) for d in s['donors']}
        if any(k in used_donor_keys for k in keys):
            continue
        if any(d3(s['xyz'], uc) < ideal * 2 for uc in used_xyz):
            continue
        final.append(s); used_xyz.append(s['xyz']); used_donor_keys.update(keys)
    return final


# --------------------------------------------------------------------------
# 7. Di-metal search: pair up mono candidates by target M-M separation,
#    optionally sharing one bridging donor
# --------------------------------------------------------------------------
def find_di_sites(donors, atoms, cn, params, cfg):
    sep_min, sep_max = cfg.sep if cfg.sep else params['sep']
    raw = find_mono_sites(donors, atoms, cn, params, cfg, dedup=False)
    raw = raw[:cfg.pair_pool]              # cap pool size for speed
    print(f"  (built {len(raw)} candidate mono-metal footprints to pair up)")

    pairs = []
    for a, b in combinations(raw, 2):
        dist = d3(a['xyz'], b['xyz'])
        if not (sep_min <= dist <= sep_max):
            continue
        keys_a = {dkey(d) for d in a['donors']}
        keys_b = {dkey(d) for d in b['donors']}
        shared = keys_a & keys_b
        if len(shared) > 1:
            continue
        if shared and not cfg.bridge:
            continue
        bonus = 0.3 if shared else 0.0     # small reward: bridging is chemically favorable
        pairs.append({'a': a, 'b': b, 'sep': dist, 'bridge': bool(shared),
                     'score': a['score'] + b['score'] - bonus})

    pairs.sort(key=lambda p: p['score'])
    used_keys, used_xyz, final = set(), [], []
    for p in pairs:
        keys = ({dkey(d) for d in p['a']['donors']} | {dkey(d) for d in p['b']['donors']})
        # reject if either metal position collides with an already-accepted one
        if any(d3(p['a']['xyz'], uc) < 3.0 or d3(p['b']['xyz'], uc) < 3.0 for uc in used_xyz):
            continue
        # reject if any donor (bridging or not) was already used by a kept pair
        if keys & used_keys:
            continue
        final.append(p)
        used_xyz += [p['a']['xyz'], p['b']['xyz']]
        used_keys.update(keys)
    return final


# --------------------------------------------------------------------------
# 8. Output
# --------------------------------------------------------------------------
def metal_line(serial, elem, resnum, xyz):
    x, y, z = xyz
    name = elem.upper().ljust(2)
    return (f"HETATM{serial:5d} {name:>2}   {elem.upper():>3} {resnum:4d}    "
           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem.upper():>2}\n")

def renumber(lines):
    n, out = 1, []
    for l in lines:
        if l[:6].strip() in ('ATOM', 'HETATM'):
            l = l[:6] + f"{n:5d}" + l[11:]
            n += 1
        out.append(l)
    return out

def dedupe_metal_positions(entries, tol=0.05):
    """entries: list of dicts with an 'xyz' key, already in the priority
    order they should be kept in (first occurrence wins). Two independent
    searches (mono, di) can converge on the IDENTICAL fitted coordinates
    for the same donor combo -- --both concatenated them with no check,
    producing exact-0.000-A duplicate metal atoms (two point charges on
    top of each other -> EM energy blows up / instant segfault).

    Returns (kept_idx, dup_of):
      kept_idx : set of entry indices that should actually be written
      dup_of   : {dropped_idx: kept_idx_it_duplicates}
    """
    kept_idx, kept_xyz, dup_of = [], [], {}
    for i, e in enumerate(entries):
        xyz = np.asarray(e['xyz'])
        match = None
        for j, kxyz in zip(kept_idx, kept_xyz):
            if np.linalg.norm(xyz - kxyz) < tol:
                match = j
                break
        if match is None:
            kept_idx.append(i)
            kept_xyz.append(xyz)
        else:
            dup_of[i] = match
    return set(kept_idx), dup_of

def donor_tag(d):
    return f"{d['chain']}:{d['resname']}{d['resnum']}-{d['atname']}"

def write_mono_pdb(path, header, atom_lines, sites, metal, cn):
    rem = [f"REMARK MetalBind mono  metal={metal} CN={cn} sites={len(sites)}\n"]
    hets = []
    for i, s in enumerate(sites, 1):
        rem.append(f"REMARK {metal}{i:02d}: CN={s['cn']} " + "|".join(donor_tag(d) for d in s['donors'])
                  + "  dists=" + "/".join(f"{v:.2f}" for v in s['dists'])
                  + f"A  score={s['score']:.3f}\n")
        hets.append(metal_line(0, metal, 9000 + i, s['xyz']))
    open(path, 'w').writelines(renumber(header + rem + atom_lines + hets + ['TER\n', 'END\n']))

def write_di_pdb(path, header, atom_lines, pairs, metal, cn):
    rem = [f"REMARK MetalBind di  metal={metal} CN={cn} pairs={len(pairs)}\n"]
    hets = []
    for i, p in enumerate(pairs, 1):
        rem.append(f"REMARK Pair{i:02d}: sep={p['sep']:.2f}A bridge={p['bridge']}\n")
        rem.append(f"REMARK   {metal}{i:02d}A: " + "|".join(donor_tag(d) for d in p['a']['donors']) + "\n")
        rem.append(f"REMARK   {metal}{i:02d}B: " + "|".join(donor_tag(d) for d in p['b']['donors']) + "\n")
        hets.append(metal_line(0, metal, 9000 + 2 * i - 1, p['a']['xyz']))
        hets.append(metal_line(0, metal, 9000 + 2 * i, p['b']['xyz']))
    open(path, 'w').writelines(renumber(header + rem + atom_lines + hets + ['TER\n', 'END\n']))

def write_both_pdb(path, header, atom_lines, sites, pairs, metal, cn, dedup_tol=0.05):
    # Flat, priority-ordered list of every metal position --both would
    # otherwise write: mono sites first, then each di-pair's two atoms.
    entries = []
    for i, s in enumerate(sites):
        entries.append({'xyz': s['xyz']})
    for i, p in enumerate(pairs):
        entries.append({'xyz': p['a']['xyz']})
        entries.append({'xyz': p['b']['xyz']})

    kept_idx, dup_of = dedupe_metal_positions(entries, tol=dedup_tol)
    n_dropped = len(entries) - len(kept_idx)

    rem = [f"REMARK MetalBind both  metal={metal} CN={cn} mono_sites={len(sites)} di_pairs={len(pairs)}\n"]
    if n_dropped:
        rem.append(f"REMARK   {n_dropped} duplicate metal position(s) removed "
                   f"(< {dedup_tol:.2f} A from another kept site) -- mono and di "
                   f"searches independently converged on the same coordinates\n")

    # Assign resnums only to entries that are actually kept/written, so
    # REMARKs can point at the real resnum of the site a duplicate merged into.
    hets, resnum, entry_resnum = [], 9000, {}
    for i in sorted(kept_idx):
        resnum += 1
        entry_resnum[i] = resnum
        hets.append(metal_line(0, metal, resnum, entries[i]['xyz']))

    idx = 0
    for i, s in enumerate(sites, 1):
        e = idx; idx += 1
        status = "" if e in kept_idx else f"  [DUPLICATE of resnum {entry_resnum[dup_of[e]]} -- not written]"
        rem.append(f"REMARK MONO {metal}{i:02d}: CN={s['cn']} " + "|".join(donor_tag(d) for d in s['donors'])
                  + "  dists=" + "/".join(f"{v:.2f}" for v in s['dists'])
                  + f"A  score={s['score']:.3f}{status}\n")

    for i, p in enumerate(pairs, 1):
        e_a, e_b = idx, idx + 1; idx += 2
        rem.append(f"REMARK DI Pair{i:02d}: sep={p['sep']:.2f}A bridge={p['bridge']}\n")
        status_a = "" if e_a in kept_idx else f"  [DUPLICATE of resnum {entry_resnum[dup_of[e_a]]} -- not written]"
        status_b = "" if e_b in kept_idx else f"  [DUPLICATE of resnum {entry_resnum[dup_of[e_b]]} -- not written]"
        rem.append(f"REMARK   {metal}{i:02d}A: " + "|".join(donor_tag(d) for d in p['a']['donors']) + status_a + "\n")
        rem.append(f"REMARK   {metal}{i:02d}B: " + "|".join(donor_tag(d) for d in p['b']['donors']) + status_b + "\n")

    open(path, 'w').writelines(renumber(header + rem + atom_lines + hets + ['TER\n', 'END\n']))
    print(f"  De-duplication: {len(entries)} candidate metal position(s) -> "
          f"{len(kept_idx)} unique atom(s) written"
          + (f"  ({n_dropped} duplicate(s) removed)" if n_dropped else "") + "\n")

def print_mono_summary(sites, metal, cn, top):
    print(f"\n  Mono-{metal} sites found (CN={cn}): {len(sites)}  (all {len(sites)} written to PDB)")
    print(f"  {'Rank':<5}{'CN':<4}{'Donors':<55}{'Dists (A)':<28}{'Score':<8}")
    print(f"  {'-'*99}")
    for i, s in enumerate(sites[:top], 1):
        dtxt = ", ".join(donor_tag(d) for d in s['donors'])
        ztxt = "/".join(f"{v:.2f}" for v in s['dists'])
        print(f"  {i:<5}{s['cn']:<4}{dtxt:<55}{ztxt:<28}{s['score']:.3f}")
    if len(sites) > top:
        print(f"  ... ({len(sites)-top} more in the PDB, not shown here; use --top to print more)")
    print()

def print_di_summary(pairs, metal, cn, top):
    print(f"\n  Di-{metal} pairs found (CN={cn} each): {len(pairs)}  (all {len(pairs)} written to PDB)")
    print(f"  {'Rank':<5}{'M-M (A)':<9}{'Bridge':<8}{'Score':<8}")
    print(f"  {'-'*40}")
    for i, p in enumerate(pairs[:top], 1):
        print(f"  {i:<5}{p['sep']:<9.2f}{str(p['bridge']):<8}{p['score']:.3f}")
        print(f"        A: " + ", ".join(donor_tag(d) for d in p['a']['donors']))
        print(f"        B: " + ", ".join(donor_tag(d) for d in p['b']['donors']))
    if len(pairs) > top:
        print(f"  ... ({len(pairs)-top} more in the PDB, not shown here; use --top to print more)")
    print()


# --------------------------------------------------------------------------
# 9. CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(prog='MetalBind.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument('input')
    ap.add_argument('--mono', metavar='METAL+CN')
    ap.add_argument('--di', metavar='METAL+CN')
    ap.add_argument('--both', metavar='METAL+CN', help='run --mono and --di with the same METAL+CN')
    cn_mode = ap.add_mutually_exclusive_group()
    cn_mode.add_argument('--only', action='store_true', help='require exactly CN donors (default)')
    cn_mode.add_argument('--upto', action='store_true', help='allow sites with 1..CN donors, not just CN')
    ap.add_argument('--don-cutoff', type=float, default=7.0, dest='don_cutoff', metavar='A')
    ap.add_argument('--clash', type=float, default=1.8, metavar='A')
    ap.add_argument('--max-dist', type=float, default=0.6, dest='max_dist', metavar='A',
                    help='tolerance added to ideal M-donor bond length for the outer M-N cutoff [0.6]')
    ap.add_argument('--sep', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'))
    ap.add_argument('--no-bridge', action='store_false', dest='bridge')
    ap.add_argument('--no-nterm', action='store_false', dest='nterm')
    ap.add_argument('--max-neighbors', type=int, default=10, dest='max_neighbors')
    ap.add_argument('--pair-pool', type=int, default=400, dest='pair_pool',
                    help='(--di) how many top mono candidates to pair up [400]')
    ap.add_argument('--dedup-tol', type=float, default=0.05, dest='dedup_tol', metavar='A',
                    help='(--both) merge/drop duplicate metal positions closer than this [0.05]')
    ap.add_argument('--top', type=int, default=5,
                    help='how many ranked sites/pairs to print to screen [5] -- the output PDB always contains ALL sites/pairs found')
    cfg = ap.parse_args()

    n_given = sum(bool(x) for x in (cfg.mono, cfg.di, cfg.both))
    if n_given != 1:
        ap.error("Specify exactly one of --mono, --di, or --both METAL+CN")
    modes = ['mono', 'di'] if cfg.both else (['mono'] if cfg.mono else ['di'])
    token = cfg.mono or cfg.di or cfg.both

    atoms, atom_lines, header = parse_pdb(cfg.input)
    donors = get_donors(atoms, include_nterm=cfg.nterm)
    n_his = sum(1 for d in donors if d['role'] != 'N-term')
    n_nterm = len(donors) - n_his
    print(f"  Parsed {len(atoms)} atoms | donors: {n_his} His-N + {n_nterm} N-term = {len(donors)} total\n")

    stem = cfg.input[:-4] if cfg.input.lower().endswith('.pdb') else cfg.input
    metal, cn, sites, pairs = None, None, [], []

    for mode in modes:
        metal, cn, params = analyze_request(token, mode)
        need = 1 if cfg.upto else cn
        if len(donors) < need:
            print(f"  [!] Skipping {mode}: need >= {need} donors, only {len(donors)} available\n")
            continue

        if mode == 'mono':
            sites = find_mono_sites(donors, atoms, cn, params, cfg, dedup=True)
            if not sites:
                print("  [!] No valid mono-metal sites found. Try --don-cutoff 9 or --max-neighbors 15\n")
                continue
            print_mono_summary(sites, metal, cn, cfg.top)
            if not cfg.both:
                out = f"{stem}_{metal}{cn}_mono.pdb"
                write_mono_pdb(out, header, atom_lines, sites, metal, cn)
                print(f"  Output: {out}\n")
        else:
            pairs = find_di_sites(donors, atoms, cn, params, cfg)
            if not pairs:
                print("  [!] No valid di-metal pairs found. Try --sep wider or --pair-pool larger\n")
                continue
            print_di_summary(pairs, metal, cn, cfg.top)
            if not cfg.both:
                out = f"{stem}_{metal}{cn}_di.pdb"
                write_di_pdb(out, header, atom_lines, pairs, metal, cn)
                print(f"  Output: {out}\n")

    if cfg.both:
        if not sites and not pairs:
            sys.exit("  [!] Nothing found for --both (no mono sites and no di pairs).")
        out = f"{stem}_{metal}{cn}_both.pdb"
        write_both_pdb(out, header, atom_lines, sites, pairs, metal, cn, dedup_tol=cfg.dedup_tol)
        print(f"  Output: {out}  ({len(sites)} mono sites + {len(pairs)} di pairs in one PDB)\n")

if __name__ == '__main__':
    main()