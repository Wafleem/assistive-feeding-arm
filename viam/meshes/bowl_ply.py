"""Write an open-top bowl wall as an ASCII PLY (metres, as Viam's mesh loader expects).

Origin at the centre of the bowl's bottom, +z up. Only the sloped wall is meshed: the
table obstacle covers the bottom, and an open top lets planned moves put the spoon inside
(Viam treats meshes as surfaces, not solid volumes).

    python viam/meshes/bowl_ply.py viam/meshes/bowl-150x70.ply 150 100 70   # rim, bottom, height (mm)
"""
import math
import sys

out = sys.argv[1]
top_d, bottom_d, height = (float(a) for a in (sys.argv[2:5] + ['150', '100', '70'][len(sys.argv[2:5]):]))
N = 32
ring = lambda r, z: [(r*math.cos(2*math.pi*i/N), r*math.sin(2*math.pi*i/N), z) for i in range(N)]
verts = ring(bottom_d/2, 0.0) + ring(top_d/2, height)
faces = []
for i in range(N):
    j = (i+1) % N
    faces += [(i, j, N+j), (i, N+j, N+i)]
with open(out, 'w') as f:
    f.write(f'ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
    f.writelines(f'{x/1000:.6f} {y/1000:.6f} {z/1000:.6f}\n' for x, y, z in verts)
    f.writelines(f'3 {a} {b} {c}\n' for a, b, c in faces)
print(f'{out}: top {top_d} mm, bottom {bottom_d} mm, height {height} mm, {len(faces)} triangles')
