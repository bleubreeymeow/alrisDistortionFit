import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import math
import xrayutilities as xu


crystal = xu.materials.Crystal.fromCIF('C:/Users/User/Downloads/subgroup.cif')
for a, n, c, d in crystal.lattice.base():
    print([a, [n[0], n[1], n[2]]])



# loop over both crystals
for c1, c2 in zip(crystal.lattice.base(), crystal_mod_o1_2_dz.lattice.base()):
    print(f"c1: {c1}")
    print(f"c2: {c2}")
