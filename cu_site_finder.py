#!/usr/bin/env python3
"""
cu_site_finder.py - predict copper-binding sites in a peptide / protein structure.

Method
------
1. Donor perception. Ligand atoms are recognised from the covalent bond graph
   rebuilt from the coordinates, not from residue-name conventions:
     N_imid       His imidazole N (free if no H attached)
     N_amine      N-terminal amine            (needs deprotonation if NH3+)
     N_amide      backbone amide N            (--amide; needs deprotonation)
     O_carboxyl   Asp / Glu / C-terminal carboxylate O
     O_carbonyl   backbone / Asn / Gln carbonyl O (weak, mostly axial)
     S_thiolate   Cys SG,  S_thioether  Met SD
2. Lone-pair direction. Each donor gets a unit vector pointing away from its
   heavy-atom neighbours and a target angle alpha between that vector and the
   donor->metal bond (0 deg for sp2 N: the metal lies on the external C-N-C
   bisector in the ring plane; ~70 deg for an amine; ~40 deg for C=O).
3. Side-chain flexibility (--flex). Each His ring is also tried in rotated
   conformers (chi1 +-15/30 deg, chi2 +-20/40 deg, ring flip); conformers that
   clash with the rest of the structure are discarded, the others carry a
   Gaussian penalty on the rotation.
4. Feasibility shells. For each donor the allowed metal positions (sphere of
   radius r0, inside the lone-pair cone, free of steric conflict) are sampled.
   Donors can share one metal only if their shells intersect, so sites are
   grown as cliques of mutually compatible donors and dead branches are pruned.
5. Refinement. The metal is placed by minimising bond-length, lone-pair-angle
   and steric penalties (Nelder-Mead, 3 variables).
6. Scoring.   score = g * sum_i  w_i * p_i * q_i
     q_i  bond-length x lone-pair-angle Gaussian quality      (0..1)
     w_i  intrinsic donor strength towards Cu (His N = 1.0, C=O = 0.35)
     p_i  penalty for a protonation change (tautomer 0.85, HIP 0.6) and for
          side-chain rotation (--flex)
     g    fit of the L-M-L angles to the preferred polyhedron, treated as a
          fragment so that positions left for water are allowed
          (Cu(II): square planar / square pyramidal / trigonal bipyramidal /
          tetrahedral, as in type-1 and type-3 protein sites; Cu(I): linear,
          trigonal planar, tetrahedral)
   Tier  A: >=3 N/S donors and g >= 0.6   B: >=2 N/S donors   C: other.
7. Selection. Greedy non-maximum suppression: each donor group (a side chain,
   or a single backbone atom) serves one metal.

Limits: backbone is rigid, so sites needing larger rearrangements are missed;
solvent and ions in the input are ignored; parameters are typical values from
structural surveys, meant for ranking and not for binding free energies.
Metal atoms already in the input are used as a reference: the program reports
how close the nearest prediction comes to each of them.

Usage
-----
    python cu_site_finder.py input.pdb [--ox 2|1] [--flex] [--top 20]
           [--min-donors 2] [--max-donors 4] [--amide] [--tautomers]
           [--no-carbonyl] [--no-termini] [--out PREFIX]
Output: PREFIX_sites.pdb (input + CU atoms, CONECT to donors, score in B,
        moved His side chains of selected sites), PREFIX_sites.csv (ranked).
"""
import argparse
import csv
import sys
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# ------------------------------------------------------------------ parameters
@dataclass(frozen=True)
class DonorClass:
    alpha: float      # target angle lone-pair vector vs donor->metal (deg)
    s_alpha: float    # tolerance on alpha (deg)
    s_r: float        # tolerance on bond length (A)
    w: float          # intrinsic donor strength towards Cu
    anchor: bool      # can anchor a site (N/S donors)
    short: str        # label used in the composition string

DONORS = {
    'N_imid':      DonorClass(0.0,  15.0, 0.15, 1.00, True,  'His'),
    'N_amine':     DonorClass(70.0, 15.0, 0.10, 0.90, True,  'NH2'),
    'N_amide':     DonorClass(0.0,  15.0, 0.10, 0.80, True,  'N-'),
    'O_carboxyl':  DonorClass(55.0, 20.0, 0.15, 0.60, False, 'COO'),
    'O_carbonyl':  DonorClass(40.0, 25.0, 0.15, 0.35, False, 'C=O'),
    'S_thiolate':  DonorClass(75.0, 15.0, 0.10, 1.00, True,  'Cys'),
    'S_thioether': DonorClass(50.0, 20.0, 0.15, 0.50, True,  'Met'),
}

# Ideal metal-donor distances (A); None = donor not used for that metal.
METALS = {
    2: dict(label='Cu(II)', polyhedra=('square planar', 'square pyramidal',
                                       'trigonal bipyramidal', 'tetrahedral'),
            r0=dict(N_imid=2.00, N_amine=2.02, N_amide=1.95, O_carboxyl=1.97,
                    O_carbonyl=2.05, S_thiolate=2.20, S_thioether=2.45)),
    1: dict(label='Cu(I)', polyhedra=('linear', 'trigonal planar', 'tetrahedral'),
            r0=dict(N_imid=1.98, N_amine=2.05, N_amide=None, O_carboxyl=2.10,
                    O_carbonyl=2.20, S_thiolate=2.20, S_thioether=2.30)),
}

POLYHEDRA = {
    'linear':           [(0, 0, 1), (0, 0, -1)],
    'trigonal planar':  [(1, 0, 0), (-0.5, 0.866, 0), (-0.5, -0.866, 0)],
    'tetrahedral':      [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
    'square planar':    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)],
    'square pyramidal': [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
    'trigonal bipyramidal': [(1, 0, 0), (-0.5, 0.866, 0), (-0.5, -0.866, 0), (0, 0, 1), (0, 0, -1)],
}

COV_RADIUS = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'SE': 1.20, 'P': 1.07}
HIS_NAMES = {'HIS', 'HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP'}
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'HN', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3'}
SOLVENT = {'HOH', 'WAT', 'TIP', 'TIP3', 'SOL', 'DOD', 'H2O'}
METAL_ELEMENTS = {'CU', 'ZN', 'FE', 'NI', 'CO', 'MN', 'CD', 'HG', 'MG', 'CA'}

CHI1_STEPS, S_CHI1 = (-30, -15, 0, 15, 30), 30.0           # deg (--flex)
CHI2_STEPS, S_CHI2 = (-40, -20, 0, 20, 40), 40.0
FLIP_PENALTY = 0.8
SHELL_DIRS = 800        # directions sampled on each donor's shell
SHELL_TOL = 0.8         # A, two shells "intersect" if points are this close
CLASH_SHELL = dict(heavy=2.4, ligand=1.8, H=1.6)  # shell sampling
CLASH_SOFT = dict(heavy=2.7, ligand=2.3, H=2.1)   # refinement penalty onset
CLASH_HARD = dict(heavy=2.5, ligand=2.1, H=1.8)   # rejection after refinement
CLASH_SC = dict(heavy=2.6, H=1.8)                 # moved side chain vs rest
MAX_DR, MAX_DA = 0.35, 2.5                        # bond (A), angle (x s_alpha)
SIGMA_GEOM = 20.0       # deg, polyhedron-fit tolerance
MIN_METAL_SEP = 2.5     # A, between accepted sites


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rotate(points, origin, axis, deg):
    """Rodrigues rotation of points about the line origin + t*axis."""
    k, t, p = unit(axis), np.radians(deg), points - origin
    return origin + p * np.cos(t) + np.cross(k, p) * np.sin(t) + np.outer(p @ k, k) * (1 - np.cos(t))


def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    phi, z = np.pi * (1 + 5 ** 0.5) * k, 1 - 2 * k / n
    rho = np.sqrt(1 - z * z)
    return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])


REF_ANGLES = {name: np.degrees(np.arccos(np.clip(unit(v) @ unit(v).T, -1, 1)))
              for name, v in POLYHEDRA.items()}
DIRS = fibonacci_sphere(SHELL_DIRS)

# ------------------------------------------------------------------- structure
@dataclass
class Structure:
    lines: list
    line_of: np.ndarray       # atom index -> position in lines
    serial: np.ndarray
    name: np.ndarray
    resn: np.ndarray
    chain: np.ndarray
    resi: np.ndarray
    icode: np.ndarray
    elem: np.ndarray
    xyz: np.ndarray
    ref_metals: list
    nbrs: list = field(default_factory=list)
    residues: dict = field(default_factory=dict)

    def tag(self, i):
        return f"{self.chain[i]}:{self.resn[i]}{self.resi[i]}{self.icode[i].strip()}.{self.name[i]}"

    def res_key(self, i):
        return self.chain[i], self.resi[i], self.icode[i]


def guess_element(line):
    e = line[76:78].strip().upper()
    return e or line[12:16].strip().lstrip('0123456789')[:1].upper()


def read_pdb(path):
    keep, rows, metals = [], [], []
    for line in open(path):
        rec = line[:6]
        if rec.startswith('ENDMDL'):
            break
        if rec.startswith('TER'):
            keep.append(line)
        if not rec.startswith(('ATOM', 'HETATM')) or line[16] not in ' A1':
            continue
        keep.append(line)
        resn, elem = line[17:21].strip(), guess_element(line)
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if elem in METAL_ELEMENTS and rec.startswith('HETATM'):
            metals.append((f"{resn}{line[22:26].strip()}{line[21]}", np.array(xyz)))
            continue
        if resn in SOLVENT:
            continue
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = -1
        rows.append((len(keep) - 1, serial, line[12:16].strip(), resn, line[21],
                     int(line[22:26]), line[26], elem, xyz))
    if not rows:
        sys.exit(f"[error] no protein atoms in {path}")
    cols = list(zip(*rows))
    st = Structure(keep, *(np.array(c) for c in cols[:8]), np.array(cols[8], float), metals)
    st.nbrs = bond_graph(st)
    for i in range(len(st.elem)):
        st.residues.setdefault(st.res_key(i), {})[st.name[i]] = i
    return st


def bond_graph(st):
    rad = np.array([COV_RADIUS.get(e, 0.80) for e in st.elem])
    pairs = cKDTree(st.xyz).query_pairs(2.2, output_type='ndarray')
    i, j = pairs.T
    d = np.linalg.norm(st.xyz[i] - st.xyz[j], axis=1)
    ok = (d < rad[i] + rad[j] + 0.40) & ~((st.elem[i] == 'H') & (st.elem[j] == 'H'))
    nbrs = [[] for _ in st.elem]
    for a, b in pairs[ok]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs

# ---------------------------------------------------------------------- donors
@dataclass
class Donor:
    idx: int
    cls: str
    xyz: np.ndarray
    v: np.ndarray            # lone-pair direction
    r0: float
    p: float                 # protonation / rotation penalty factor
    group: tuple             # exclusive unit: side chain or backbone atom
    exempt: set              # atoms ignored in clash tests (donor, bonded atoms)
    notes: list
    moved: dict = field(default_factory=dict)   # atom index -> new xyz (--flex)
    shell: np.ndarray = None

    @property
    def dc(self):
        return DONORS[self.cls]


def classify(st, i):
    """Return (class, protonation penalty, notes) for atom i, or None."""
    e, name, resn = st.elem[i], st.name[i], st.resn[i]
    heavy = [j for j in st.nbrs[i] if st.elem[j] != 'H']
    hyd = [j for j in st.nbrs[i] if st.elem[j] == 'H']
    if e == 'N' and resn in HIS_NAMES and name in ('ND1', 'NE2'):
        if not hyd:
            return 'N_imid', 1.0, []
        other = st.residues[st.res_key(i)].get('NE2' if name == 'ND1' else 'ND1')
        if other is not None and any(st.elem[k] == 'H' for k in st.nbrs[other]):
            return 'N_imid', 0.6, ['HIP deprotonation']
        return 'N_imid', 0.85, ['tautomer switch']
    if e == 'N' and name == 'N' and resn != 'PRO':
        if len(heavy) == 1:
            return 'N_amine', 1.0, ['N-terminus'] + (['deprotonation'] if len(hyd) >= 3 else [])
        return 'N_amide', 1.0, ['amide deprotonation']
    if e == 'O' and len(heavy) == 1 and not hyd and st.elem[heavy[0]] == 'C':
        c = heavy[0]
        if np.linalg.norm(st.xyz[c] - st.xyz[i]) > 1.31:      # C-O single bond
            return None
        if sum(st.elem[j] == 'O' for j in st.nbrs[c]) == 2:
            return 'O_carboxyl', 1.0, (['C-terminus'] if name in ('O', 'OXT') else [])
        if sum(st.elem[j] != 'H' for j in st.nbrs[c]) == 3:
            return 'O_carbonyl', 1.0, []
        return None
    if e == 'S' and not any(st.elem[j] == 'S' for j in heavy):
        if len(heavy) == 1:
            return 'S_thiolate', 1.0, (['thiol deprotonation'] if hyd else [])
        if len(heavy) == 2:
            return 'S_thioether', 1.0, []
    return None


def lone_pair(st, i, xyz_of):
    heavy = [j for j in st.nbrs[i] if st.elem[j] != 'H']
    return unit(sum(unit(xyz_of(i) - xyz_of(j)) for j in heavy)) if heavy else None


def his_conformers(st, i, tree):
    """Rotated His side chains (chi1, chi2, ring flip) free of steric clashes."""
    res = st.residues[st.res_key(i)]
    if not all(a in res for a in ('CA', 'CB', 'CG')):
        return
    ca, cb, cg = (st.xyz[res[a]] for a in ('CA', 'CB', 'CG'))
    sc = [k for n, k in res.items() if n not in BACKBONE and n != 'CB']   # beyond CB
    ring = [k for k in sc if st.name[k] != 'CG' and not st.name[k].startswith('HB')]
    own = set(sc) | {res['CA'], res['CB']}
    for c1 in CHI1_STEPS:
        for c2 in CHI2_STEPS:
            for flip in (False, True):
                if c1 == c2 == 0 and not flip:
                    continue
                x = rotate(st.xyz[sc], cb, cb - ca, c1)
                pos = dict(zip(sc, x))
                cg_new = pos[res['CG']]
                x_ring = rotate(np.array([pos[k] for k in ring]), cb, cg_new - cb, c2 + 180 * flip)
                pos.update(zip(ring, x_ring))
                if not clash_free(st, pos, own, tree):
                    continue
                pen = np.exp(-0.5 * ((c1 / S_CHI1) ** 2 + (c2 / S_CHI2) ** 2)) * (FLIP_PENALTY if flip else 1.0)
                label = f"chi1 {c1:+d} chi2 {c2:+d}" + (" flip" if flip else "")
                yield label, pen, pos


def clash_free(st, pos, own, tree):
    for k, x in pos.items():
        for j in tree.query_ball_point(x, CLASH_SC['heavy']):
            if j in own:
                continue
            lim = CLASH_SC['H'] if 'H' in (st.elem[j], st.elem[k]) else CLASH_SC['heavy']
            if np.linalg.norm(st.xyz[j] - x) < lim:
                return False
    return True


def perceive_donors(st, metal, opt, tree):
    """All ligand-capable atoms (used for clash rules) and the enabled donor pool."""
    res_id, capable, pool = {}, set(), []
    for i in range(len(st.elem)):
        if st.elem[i] not in ('N', 'O', 'S'):
            continue
        got = classify(st, i)
        if got is None:
            continue
        cls, p, notes = got
        capable.add(i)
        r0 = metal['r0'].get(cls)
        if r0 is None or (p < 1.0 and not opt.tautomers):
            continue
        if (cls == 'N_amide' and not opt.amide) or (cls == 'O_carbonyl' and not opt.carbonyl):
            continue
        if opt.no_termini and ('N-terminus' in notes or 'C-terminus' in notes):
            continue
        v = lone_pair(st, i, lambda k: st.xyz[k])
        if v is None:
            continue
        rkey = res_id.setdefault(st.res_key(i), len(res_id))
        group = (rkey, st.name[i]) if st.name[i] in BACKBONE else (rkey, 'sc')
        exempt = {i, *st.nbrs[i]}
        pool.append(Donor(i, cls, st.xyz[i], v, r0, p, group, exempt, notes))
        if opt.flex and cls == 'N_imid':
            for label, pen, pos in his_conformers(st, i, tree):
                v = lone_pair(st, i, lambda k: pos.get(k, st.xyz[k]))
                pool.append(Donor(i, cls, pos[i], v, r0, p * pen, group, exempt, notes + [label], pos))
    return capable, pool

# ---------------------------------------------------------- shells and cliques
def environment(st, donors, tree, center, radius):
    """Atoms around center as (index array, xyz array), with the donors' moved
    side chains in place of their original coordinates; donor-exempt atoms out."""
    moved = {k: x for d in donors for k, x in d.moved.items()}
    exempt = set().union(*(d.exempt for d in donors))
    idx = [k for k in tree.query_ball_point(center, radius) if k not in exempt and k not in moved]
    extra = [k for k, x in moved.items() if k not in exempt and np.linalg.norm(x - center) < radius]
    xyz = np.vstack([st.xyz[idx].reshape(-1, 3), np.array([moved[k] for k in extra]).reshape(-1, 3)])
    return np.array(idx + extra, int), xyz


def radii(st, capable, idx, table):
    return np.array([table['H'] if st.elem[k] == 'H' else
                     table['ligand'] if k in capable else table['heavy'] for k in idx])


def build_shells(st, capable, pool, tree):
    """Sample each donor's allowed metal positions; drop inaccessible donors."""
    kept = []
    for d in pool:
        theta = np.degrees(np.arccos(np.clip(DIRS @ d.v, -1, 1)))
        pts = d.xyz + d.r0 * DIRS[np.abs(theta - d.dc.alpha) <= 2 * d.dc.s_alpha]
        idx, env = environment(st, [d], tree, d.xyz, d.r0 + CLASH_SHELL['heavy'])
        if len(idx):
            dist = np.linalg.norm(pts[:, None] - env[None], axis=2)
            pts = pts[(dist >= radii(st, capable, idx, CLASH_SHELL)).all(axis=1)]
        if len(pts):
            d.shell = pts
            kept.append(d)
    return kept


def enumerate_sites(pool, min_d, max_d):
    """Yield (members, feasible points) for cliques of compatible donors."""
    trees = [cKDTree(d.shell) for d in pool]
    reach = 2 * max(d.r0 for d in pool) + SHELL_TOL
    pos = cKDTree(np.array([d.xyz for d in pool]))
    adj = [set() for _ in pool]
    for i, j in pos.query_pairs(reach):
        a, b = pool[i], pool[j]
        if a.group[0] == b.group[0] and (a.group == b.group or (a.moved and b.moved)):
            continue                     # same side chain / two conformers of one residue
        if np.isfinite(trees[j].query(a.shell, distance_upper_bound=SHELL_TOL)[0]).any():
            adj[i].add(j)
            adj[j].add(i)

    def grow(members, pts, cands):
        if len(members) >= min_d and any(pool[m].dc.anchor for m in members):
            yield members, pts
        if len(members) == max_d:
            return
        for j in cands:
            d, _ = trees[j].query(pts, distance_upper_bound=SHELL_TOL)
            sub = pts[np.isfinite(d)]
            if len(sub):
                yield from grow(members + [j], sub, [k for k in cands if k > j and k in adj[j]])

    for i in range(len(pool)):
        yield from grow([i], pool[i].shell, sorted(k for k in adj[i] if k > i))

# ------------------------------------------------------------ refine and score
@dataclass
class Site:
    xyz: np.ndarray
    donors: list
    r: np.ndarray
    dev: np.ndarray
    q: np.ndarray
    score: float
    strength: float
    geometry: str
    rms: float
    g: float
    tau4: float
    contacts: list
    burial: int
    chains: str
    tier: str = ''


def polyhedron_fit(u, names):
    """Smallest RMS deviation (deg) of observed L-M-L angles from any assignment
    of the ligands to vertices of the ideal polyhedra (fragment fit)."""
    n = len(u)
    if n < 2:
        return 0.0, names[0]
    obs = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
    iu = np.triu_indices(n, 1)
    fits = {name: min(np.sqrt(np.mean((obs[iu] - REF_ANGLES[name][np.ix_(p, p)][iu]) ** 2))
                      for p in permutations(range(len(REF_ANGLES[name])), n))
            for name in names if n <= len(REF_ANGLES[name])}
    if not fits:
        return 180.0, 'none'
    best = min(fits.values())
    name = next(nm for nm in names if nm in fits and fits[nm] <= best + 1.0)
    return fits[name], name


def tau4(u):
    if len(u) != 4:
        return float('nan')
    ang = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))[np.triu_indices(4, 1)]
    a, b = np.sort(ang)[-2:]
    return (360.0 - a - b) / 141.0


def sidechains_compatible(st, members):
    """Moved side chains of different members must not overlap each other."""
    for a, b in combinations([d for d in members if d.moved], 2):
        xa, xb = np.array(list(a.moved.values())), np.array(list(b.moved.values()))
        if np.linalg.norm(xa[:, None] - xb[None], axis=2).min() < CLASH_SC['H']:
            return False
    return True


def refine(members, start, st, capable, tree, metal):
    if not sidechains_compatible(st, members):
        return None
    D = np.array([d.xyz for d in members])
    V = np.array([d.v for d in members])
    r0 = np.array([d.r0 for d in members])
    sr = np.array([d.dc.s_r for d in members])
    al = np.radians([d.dc.alpha for d in members])
    sa = np.radians([d.dc.s_alpha for d in members])
    w = np.array([d.dc.w for d in members])
    env_idx, E = environment(st, members, tree, start, 7.0)
    soft = radii(st, capable, env_idx, CLASH_SOFT)

    def geometry(m):
        b = m - D
        r = np.linalg.norm(b, axis=1)
        return r, np.arccos(np.clip((b * V).sum(1) / r, -1, 1))

    def energy(m):
        r, th = geometry(m)
        e = (w * (((r - r0) / sr) ** 2 + ((th - al) / sa) ** 2)).sum() / w.sum()
        if len(E):
            over = np.clip(soft - np.linalg.norm(E - m, axis=1), 0, None)
            e += ((over / 0.1) ** 2).sum()
        return e

    m = minimize(energy, start, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=1e-5, maxiter=800)).x
    r, th = geometry(m)
    dev = np.degrees(th - al)
    if (np.abs(r - r0) > MAX_DR).any() or (np.abs(dev) > MAX_DA * np.degrees(sa)).any():
        return None
    de = np.linalg.norm(E - m, axis=1)
    if len(E) and (de < radii(st, capable, env_idx, CLASH_HARD)).any():
        return None

    q = np.exp(-0.5 * ((r - r0) / sr) ** 2) * np.exp(-0.5 * (np.radians(dev) / sa) ** 2)
    p = np.array([d.p for d in members])
    u = unit(D - m)
    rms, poly = polyhedron_fit(u, metal['polyhedra'])
    g = float(np.exp(-0.5 * (rms / SIGMA_GEOM) ** 2))
    strength = float((w * p * q).sum())
    member_atoms = {d.idx for d in members}
    contacts = sorted({(int(k), round(float(x), 2)) for k, x in zip(env_idx, de)
                       if k in capable and k not in member_atoms and x < 2.9})
    burial = int(sum(st.elem[k] != 'H' for k in tree.query_ball_point(m, 8.0)))
    site = Site(m, members, r, dev, q, g * strength, strength, poly, rms, g, tau4(u),
                contacts, burial, ''.join(sorted({st.chain[d.idx] for d in members})))
    n_strong = sum(d.dc.anchor and qi >= 0.5 for d, qi in zip(members, q))
    site.tier = 'A' if n_strong >= 3 and g >= 0.6 else 'B' if n_strong >= 2 else 'C'
    return site


def select(sites):
    """Greedy non-maximum suppression: one metal per donor group; moved side
    chains of accepted sites must not clash with later ones."""
    used, kept, moved = set(), [], {}
    for s in sorted(sites, key=lambda s: -s.score):
        groups = {d.group for d in s.donors}
        if groups & used or any(np.linalg.norm(s.xyz - k.xyz) < MIN_METAL_SEP for k in kept):
            continue
        new = {k: x for d in s.donors for k, x in d.moved.items()}
        if new and (moved or kept):
            a = np.array(list(new.values()))
            b = np.array(list(moved.values()) + [k.xyz for k in kept])
            if np.linalg.norm(a[:, None] - b[None], axis=2).min() < CLASH_SC['H']:
                continue
        if moved and np.linalg.norm(np.array(list(moved.values())) - s.xyz, axis=1).min() < CLASH_SC['H']:
            continue
        kept.append(s)
        used |= groups
        moved.update(new)
    return kept

# ---------------------------------------------------------------------- output
def composition(site):
    counts = {}
    for d in site.donors:
        counts[d.dc.short] = counts.get(d.dc.short, 0) + 1
    return '+'.join(f"{n}{k}" for k, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def donor_text(st, site):
    return '; '.join(f"{st.tag(d.idx)} {r:.2f}A {dv:+.0f}deg" for d, r, dv in
                     zip(site.donors, site.r, site.dev))


def notes_text(st, site):
    notes = [f"{st.tag(d.idx)} {n}" for d in site.donors for n in d.notes]
    notes += [f"{st.tag(k)} weak/axial {x:.2f}A" for k, x in site.contacts]
    return '; '.join(notes) or '-'


def write_csv(path, st, sites):
    head = ['rank', 'tier', 'score', 'strength', 'geometry', 'geom_rms_deg', 'tau4',
            'n_donors', 'composition', 'chains', 'burial_8A', 'x', 'y', 'z', 'donors', 'notes']
    with open(path, 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(head)
        for k, s in enumerate(sites, 1):
            out.writerow([
                k, s.tier, f"{s.score:.3f}", f"{s.strength:.3f}", s.geometry, f"{s.rms:.1f}",
                '' if np.isnan(s.tau4) else f"{s.tau4:.2f}", len(s.donors), composition(s), s.chains, s.burial,
                *(f"{c:.3f}" for c in s.xyz), donor_text(st, s), notes_text(st, s)])


def write_pdb(path, st, sites, metal, args):
    lines = list(st.lines)
    for s in sites:                                   # apply moved side chains
        for d in s.donors:
            for k, (x, y, z) in d.moved.items():
                l = lines[st.line_of[k]]
                lines[st.line_of[k]] = f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}"
    chain = next((c for c in 'MXYZmxyz0123456789' if c not in set(st.chain)), ' ')
    serial = max(int(st.serial.max()), 0)
    rem = [f"REMARK 250 cu_site_finder: {metal['label']}, {len(sites)} sites, donors "
           f"{args.min_donors}-{args.max_donors}, flex={args.flex}, B-factor = score\n"]
    het, conect = [], []
    for k, s in enumerate(sites, 1):
        serial += 1
        x, y, z = s.xyz
        rem.append(f"REMARK 250 CU{k:03d} tier {s.tier} score {s.score:.2f} {s.geometry} "
                   f"{composition(s)} | {donor_text(st, s)}\n")
        het.append(f"HETATM{serial % 100000:5d} CU    CU {chain}{k:4d}    "
                   f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{min(s.score, 999):6.2f}          CU\n")
        if serial < 100000 and (st.serial[[d.idx for d in s.donors]] > 0).all():
            conect.append(f"CONECT{serial:5d}" + ''.join(f"{st.serial[d.idx]:5d}" for d in s.donors) + '\n')
    with open(path, 'w') as f:
        f.writelines(rem + lines + het + ['TER\n'] + conect + ['END\n'])


def print_report(st, sites, top):
    print(f"\n  {'#':>3} {'tier':4} {'score':>6} {'geom':>5}  {'composition':14} {'chains':6} donors")
    for k, s in enumerate(sites[:top], 1):
        moved = '*' if any(d.moved for d in s.donors) else ' '
        print(f"  {k:3d} {s.tier:4} {s.score:6.2f} {s.rms:5.1f}{moved} {composition(s):14} {s.chains:6} "
              f"{donor_text(st, s)}")
    if len(sites) > top:
        print(f"  ... {len(sites) - top} more in the CSV/PDB")
    print("  (* = needs a His side-chain rotation, see CSV notes)" if any(
        d.moved for s in sites for d in s.donors) else '', end='')
    tiers = {t: sum(s.tier == t for s in sites) for t in 'ABC'}
    print(f"\n  tiers: A {tiers['A']}  B {tiers['B']}  C {tiers['C']}")
    for name, ref in st.ref_metals:
        dist = [np.linalg.norm(s.xyz - ref) for s in sites]
        if dist:
            k = int(np.argmin(dist))
            print(f"  reference metal {name}: nearest prediction #{k + 1} at {dist[k]:.2f} A")

# ------------------------------------------------------------------------ main
def find_sites(st, metal, args, verbose=True):
    tree = cKDTree(st.xyz)
    capable, pool = perceive_donors(st, metal, args, tree)
    n_pool = len(pool)
    pool = build_shells(st, capable, pool, tree)
    if len(pool) < args.min_donors:
        return []
    candidates, n_tested = [], 0
    for members, pts in enumerate_sites(pool, args.min_donors, args.max_donors):
        n_tested += 1
        site = refine([pool[m] for m in members], pts.mean(axis=0), st, capable, tree, metal)
        if site is not None:
            candidates.append(site)
    sites = select(candidates)
    if verbose:
        by_cls = {c: sum(d.cls == c for d in pool) for c in DONORS}
        print(f"  donor positions: {n_pool} generated, {len(pool)} accessible  ("
              + ', '.join(f"{c} {n}" for c, n in by_cls.items() if n) + ')')
        print(f"  donor sets tested {n_tested}, valid {len(candidates)}, selected sites {len(sites)}")
    return sites


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdb')
    ap.add_argument('--ox', type=int, choices=(1, 2), default=2, help='copper oxidation state [2]')
    ap.add_argument('--flex', action='store_true', help='sample His side-chain rotations')
    ap.add_argument('--min-donors', type=int, default=2)
    ap.add_argument('--max-donors', type=int, default=4)
    ap.add_argument('--amide', action='store_true', help='allow deprotonated backbone amide N')
    ap.add_argument('--tautomers', action='store_true', help='allow His N that carry H (tautomer switch / HIP)')
    ap.add_argument('--no-carbonyl', dest='carbonyl', action='store_false', help='ignore carbonyl O donors')
    ap.add_argument('--no-termini', action='store_true', help='ignore N-terminal amine and C-terminal carboxylate')
    ap.add_argument('--top', type=int, default=20, help='rows printed to screen [20]')
    ap.add_argument('--out', help='output prefix [input stem + _Cu<ox>]')
    args = ap.parse_args()

    metal = METALS[args.ox]
    st = read_pdb(args.pdb)
    print(f"  {args.pdb}: {len(st.elem)} atoms, {len(set(st.chain))} chains, {metal['label']}, flex={args.flex}")
    sites = find_sites(st, metal, args)

    prefix = args.out or f"{Path(args.pdb).stem}_Cu{args.ox}" + ('_flex' if args.flex else '')
    write_csv(f"{prefix}_sites.csv", st, sites)
    write_pdb(f"{prefix}_sites.pdb", st, sites, metal, args)
    print_report(st, sites, args.top)
    print(f"\n  wrote {prefix}_sites.pdb and {prefix}_sites.csv")


if __name__ == '__main__':
    main()
