#!/usr/bin/env python3
import re
from pathlib import Path
from math import isclose
from collections import OrderedDict

# --- config --------------------------------------------------------------
# Atomic numbers
Z = {"Pr": 59, "O": 8, "Ni": 28, "Cu": 29, "Ba": 56}


# Skip rows whose displacements are all zero
DROP_ZERO_DISP = False

# Name blocks as M1, M2, ...
BLOCK_VAR_PREFIX = "M"

# Emit comments mapping each block header -> its variable
EMIT_BLOCK_COMMENTS = True
# ------------------------------------------------------------------------


def nice_float(x):
    s = f"{round(float(x), 8):.8f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def label_to_element(label):
    m = re.match(r"([A-Za-z]+)", label)
    if not m:
        raise ValueError(f"Cannot parse element from label: {label}")
    return m.group(1)


def coeff_term(coeff, block_var):
    """
    Turn a numeric coeff into ' ± var', ' ± 0.5*var', or ' ± mag*var'.
    Return '' if coeff == 0.
    """
    if isclose(coeff, 0.0, abs_tol=1e-12):
        return ""
    sign = "+" if coeff > 0 else "-"
    mag = abs(coeff)
    if isclose(mag, 1.0, abs_tol=1e-9):
        return f" {sign} {block_var}"
    elif isclose(mag, 0.5, abs_tol=1e-9):
        return f" {sign} 0.5*{block_var}"
    else:
        return f" {sign} {nice_float(mag)}*{block_var}"


def parse(lines):
    """
    Yield (block_index, block_header, label, [x,y,z,dx,dy,dz]).
    block_index increments at each new mode header line.
    """
    current_label = None
    current_block_header = None
    block_idx = 0

    header_re = re.compile(r".*normfactor\s*=\s*[-\d.]+")
    start_re = re.compile(
        r"^\s*([A-Za-z]+\d+_\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )
    cont_re = re.compile(
        r"^\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )

    for raw in lines:
        line = raw.rstrip("\n")

        if not line.strip():
            current_label = None
            continue

        if line.startswith("Displacive mode"):
            current_label = None
            current_block_header = None
            continue

        if header_re.match(line):
            current_block_header = line.strip()
            block_idx += 1
            current_label = None
            continue

        m = start_re.match(line)
        if m:
            current_label = m.group(1)
            nums = list(map(float, m.groups()[1:]))
            yield block_idx, current_block_header, current_label, nums
            continue

        if current_label:
            m2 = cont_re.match(line)
            if m2:
                nums = list(map(float, m2.groups()))
                yield block_idx, current_block_header, current_label, nums
                continue
        # ignore anything else


def build_res_per_block_var(path_txt):
    """
    Deduplicate atoms by (elem, Z, x0, y0, z0), but
    accumulate contributions from each block using that block's single variable.
    """
    lines = Path(path_txt).read_text(encoding="utf-8", errors="ignore").splitlines()

    atoms = OrderedDict()  # key -> dict(elem, Z, x0,y0,z0, terms=[list for x,y,z])
    block_meta = OrderedDict()  # block_idx -> (block_var, header)

    for block_idx, block_header, label, (x, y, z, dx, dy, dz) in parse(lines):
        if block_idx == 0:
            continue

        if block_idx not in block_meta:
            block_var = f"{BLOCK_VAR_PREFIX}{block_idx}"
            block_meta[block_idx] = (block_var, block_header or f"Block {block_idx}")

        if DROP_ZERO_DISP and all(isclose(v, 0.0, abs_tol=1e-12) for v in (dx, dy, dz)):
            continue

        elem = label_to_element(label)
        Znum = Z.get(elem)
        if Znum is None:
            raise ValueError(f"Atomic number for element '{elem}' not in Z mapping.")

        x0, y0, z0 = nice_float(x), nice_float(y), nice_float(z)
        key = (elem, Znum, x0, y0, z0)
        if key not in atoms:
            atoms[key] = {
                "elem": elem,
                "Z": Znum,
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "terms": [[], [], []],
            }

        block_var = block_meta[block_idx][0]

        tx = coeff_term(dx, block_var)
        ty = coeff_term(dy, block_var)
        tz = coeff_term(dz, block_var)

        if tx:
            atoms[key]["terms"][0].append(tx)
        if ty:
            atoms[key]["terms"][1].append(ty)
        if tz:
            atoms[key]["terms"][2].append(tz)

    return atoms, block_meta


def emit_python(atoms, block_meta):
    # Build function header with variables as arguments
    vars_list = ", ".join(var for var, _ in block_meta.values())
    out = []
    out.append(f"def shift_atoms({vars_list}):")
    out.append('    """')
    out.append("    Function to shift atoms in the structure.")
    out.append("    :return: Structure of shifted atoms")
    out.append('    """')

    if EMIT_BLOCK_COMMENTS and block_meta:
        out.append("    # Mode blocks (one variable per block):")
        for var, header in block_meta.values():
            out.append(f"    #   {var}: {header}")

    out.append("    res = [")

    current_elem = None
    for (elem, Znum, x0, y0, z0), payload in atoms.items():
        if current_elem is not None and elem != current_elem:
            out.append("")
        current_elem = elem

        tx_list, ty_list, tz_list = payload["terms"]
        x_expr = x0 + "".join(tx_list) if tx_list else x0
        y_expr = y0 + "".join(ty_list) if ty_list else y0
        z_expr = z0 + "".join(tz_list) if tz_list else z0

        out.append(f"        ['{elem}', {Znum}, [{x_expr}, {y_expr}, {z_expr}]],")

    out.append("    ]")
    out.append("    return res")
    return "\n".join(out)


if __name__ == "__main__":
    infile = "c:/Users/User/Desktop/uzh_intern/alrisDistortionFit/PBCO/PBCO_1_3_inplane/C4/C4_modes.txt"  # modes_PBCO_47Pmmm.txt modes_P4.txt modes_65_Cmmm.txt
    atoms, block_meta = build_res_per_block_var(infile)
    print(emit_python(atoms, block_meta))
