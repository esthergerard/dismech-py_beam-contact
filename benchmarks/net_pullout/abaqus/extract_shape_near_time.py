"""
Extract the deformed shape of the field frame CLOSEST to a given LOCAL
step time (not just the last frame -- see extract_final_shape.py for
that case), in the same label,x,y,z format.

Run with Abaqus's own python:
    abaqus python extract_shape_near_time.py <job_name> <local_step_time>
Writes <job_name>_shape_t<local_step_time>.csv next to the odb.
"""
from __future__ import print_function
import sys
from odbAccess import openOdb

job = sys.argv[1]
t_target = float(sys.argv[2])
odb = openOdb(job + ".odb", readOnly=True)
step_name = list(odb.steps.keys())[-1]
step = odb.steps[step_name]
inst = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]

if len(step.frames) == 0:
    raise SystemExit("step %s has no field frames" % step_name)

frames = step.frames
best = min(frames, key=lambda f: abs(f.frameValue - t_target))

disp = best.fieldOutputs["U"]
coords = dict((n.label, n.coordinates) for n in inst.nodes)

out = "%s_shape_t%s.csv" % (job, sys.argv[2])
with open(out, "w") as f:
    f.write("label,x,y,z\n")
    for v in disp.values:
        c = coords[v.nodeLabel]
        d = v.data
        f.write("%d,%.9e,%.9e,%.9e\n"
                % (v.nodeLabel, c[0] + d[0], c[1] + d[1], c[2] + d[2]))
print("wrote %s (job %s, step %s, requested t=%.4f, actual frame t=%.4f)"
      % (out, job, step_name, t_target, best.frameValue))
odb.close()
