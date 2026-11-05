"""
Extract the final deformed shape (last field frame) of one job's .odb
as label,x,y,z -- the format used by the shape-comparison figures
(make_paper_figure_phase_grid_alpha_variants.py).

Run with Abaqus's own python:
    abaqus python extract_final_shape.py <job_name>
Writes <job_name>_final_shape.csv next to the odb.
"""
from __future__ import print_function
import sys
from odbAccess import openOdb

job = sys.argv[1]
odb = openOdb(job + ".odb", readOnly=True)
step_name = list(odb.steps.keys())[-1]
step = odb.steps[step_name]
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]

if len(step.frames) == 0:
    raise SystemExit("step %s has no field frames" % step_name)

last = step.frames[-1]
disp = last.fieldOutputs["U"]
coords = dict((n.label, n.coordinates) for n in inst.nodes)

out = job + "_final_shape.csv"
with open(out, "w") as f:
    f.write("label,x,y,z\n")
    for v in disp.values:
        c = coords[v.nodeLabel]
        d = v.data
        f.write("%d,%.9e,%.9e,%.9e\n"
                % (v.nodeLabel, c[0] + d[0], c[1] + d[1], c[2] + d[2]))
print("wrote %s (job %s, step %s, frame t=%.4f)"
      % (out, job, step_name, last.frameValue))
odb.close()
