"""
Extract results from the 5-JOB restart chain (net_pullout_phase{A,B,C,D,E}_
*.inp, run via run_phaseA_to_E_chained.sh), stitching the 5 SEPARATE .odb
files into one continuous time axis.

Run with ABAQUS'S OWN python (regular python cannot read .odb), from the
directory containing all 5 .odb files:

    abaqus python extract_phaseA_to_E_chained_odb.py

(Optionally pass the 5 odb paths explicitly, in A..E order, if they are
not named per the default convention below.)

Writes (next to this script, using "net_pullout_phaseE_pullout" as the
output basename since that is the final/complete state of the chain):
    <base>_energies.csv      t_total [s], job, ALLKE, ALLIE, ALLWK, ALLVD, ALLMW, ETOTAL
    <base>_engage_force.csv  Phase A only
    <base>_central_clamp.csv t_total [s], job, U1_left, U1_right, F_left, F_right
                             -- the dismech-comparable trace (sign-flipped
                             RF1 so positive = axial tension pulling the
                             strand taut)
    <base>_final_shape.csv   deformed shape at the very last frame of job E

Notes
-----
- Each job's history/field output time values are LOCAL to that job's own
  odb (reset to 0); this script adds each job's cumulative duration
  (summed from the step.timePeriod of all PRECEDING jobs) to build one
  continuous t_total axis across the whole 5-job chain.
- A job can contain MULTIPLE steps only if you used RESUME_INPLACE (same
  job resubmitted after a mid-step crash) -- Abaqus appends the resumed
  portion as an additional step in the SAME odb in that case. This script
  handles that by summing over ALL steps found in each odb, not just one.
- Works with the flat (no-parts) input: Abaqus wraps everything in the
  instance 'PART-1-1' and uppercases all set names.
"""
from __future__ import print_function
import sys
import os
from odbAccess import openOdb

ENERGY_VARS = ["ALLKE", "ALLIE", "ALLWK", "ALLVD", "ALLMW", "ETOTAL"]
DEFAULT_JOBS = [
    "net_pullout_phaseA_engage",
    "net_pullout_phaseB_hold",
    "net_pullout_phaseC_release",
    "net_pullout_phaseD_free",
    "net_pullout_phaseE_pullout",
]

job_names = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_JOBS
base = "net_pullout_phaseE_pullout"   # final/complete state of the chain

odbs = []
for name in job_names:
    path = name if name.endswith(".odb") else name + ".odb"
    if not os.path.isfile(path):
        print("WARNING: %s not found -- this phase is missing/incomplete; "
              "stopping the stitched series there." % path)
        break
    odbs.append((name, openOdb(path, readOnly=True)))

if not odbs:
    raise SystemExit("No .odb files found (checked: %s)" % job_names)

# Flatten to a list of (job_name, step_object) in chronological order,
# with the cumulative start time of each step for the combined t_total axis.
all_steps = []
acc = 0.0
for job_name, odb in odbs:
    for step_name in odb.steps.keys():
        step = odb.steps[step_name]
        all_steps.append((job_name, step))
        acc += step.timePeriod
t0 = {}
acc = 0.0
for job_name, step in all_steps:
    t0[id(step)] = acc
    acc += step.timePeriod


def find_node_labels(odb, set_name):
    ra = odb.rootAssembly
    want = set_name.upper()
    for key in ra.nodeSets.keys():
        if key.upper().endswith(want):
            labs = []
            for item in ra.nodeSets[key].nodes:
                if hasattr(item, "label"):
                    labs.append(item.label)
                else:
                    labs.extend(n.label for n in item)
            return labs
    for iname in ra.instances.keys():
        nsets = ra.instances[iname].nodeSets
        for key in nsets.keys():
            if key.upper().endswith(want):
                labs = []
                for item in nsets[key].nodes:
                    if hasattr(item, "label"):
                        labs.append(item.label)
                    else:
                        labs.extend(n.label for n in item)
                return labs
    return None


# ---------------------------------------------------------------- energies
rows = []
for job_name, step in all_steps:
    # This flat/no-CAE .inp convention names the assembly-level history
    # region "Assembly Assembly-1", not "Assembly ASSEMBLY" (the latter
    # would apply to a CAE-imported model instead).
    #
    # A job that redeclares *Output, field without restating *Output,
    # history has ZERO history regions at all -- skip it gracefully with a
    # note rather than crashing, so already-complete phases still extract.
    if "Assembly Assembly-1" not in step.historyRegions.keys():
        print("  [no history output for %s/%s -- skipping energies for "
              "this phase]" % (job_name, step.name))
        continue
    asm_region = step.historyRegions["Assembly Assembly-1"]
    series = {}
    times = None
    for var in ENERGY_VARS:
        if var in asm_region.historyOutputs.keys():
            data = asm_region.historyOutputs[var].data
            series[var] = [v for _, v in data]
            times = [t for t, _ in data]
    if times is None:
        continue
    for i, t in enumerate(times):
        rows.append([t + t0[id(step)], job_name] +
                     [series[v][i] if v in series else float("nan")
                      for v in ENERGY_VARS])

with open(base + "_energies.csv", "w") as f:
    f.write("t_total,job," + ",".join(ENERGY_VARS) + "\n")
    for r in rows:
        f.write("%.9e,%s," % (r[0], r[1]) +
                ",".join("%.9e" % v for v in r[2:]) + "\n")
print("wrote %s_energies.csv  (%d rows across %d steps/%d jobs)"
      % (base, len(rows), len(all_steps), len(odbs)))

# ------------------------------------------------- Phase A per-beam forces
engage_odb = next((odb for name, odb in odbs if "ENGAGE" in odb.steps.keys()),
                   None)
if engage_odb is not None:
    step = engage_odb.steps["ENGAGE"]
    beam_of_node = {}
    for i in range(5):
        labs = find_node_labels(engage_odb, "X_BEAM_%d_END_POS" % i)
        if labs:
            for lab in labs:
                beam_of_node[lab] = i
    rf, u1_any, t_hist = {}, None, None
    for name, region in step.historyRegions.items():
        if not name.startswith("Node"):
            continue
        label = int(name.split(".")[-1])
        outs = region.historyOutputs
        if "RF1" in outs.keys():
            data = outs["RF1"].data
            rf[label] = [v for _, v in data]
            if t_hist is None:
                t_hist = [t for t, _ in data]
        if u1_any is None and "U1" in outs.keys():
            u1_any = [v for _, v in outs["U1"].data]
    if t_hist is not None:
        with open(base + "_engage_force.csv", "w") as f:
            f.write("time,u1," + ",".join("F_X_BEAM_%d" % i for i in range(5)) +
                    ",F_total\n")
            for k, t in enumerate(t_hist):
                per_beam = [0.0] * 5
                total = 0.0
                for lab, vals in rf.items():
                    total += vals[k]
                    b = beam_of_node.get(lab)
                    if b is not None:
                        per_beam[b] += vals[k]
                f.write("%.9e,%.9e," % (t, u1_any[k]) +
                        ",".join("%.9e" % v for v in per_beam) +
                        ",%.9e\n" % total)
        print("wrote %s_engage_force.csv  (%d rows)" % (base, len(t_hist)))

# ------------------------------------------- central clamp trace (A -> E)
neg_labels = find_node_labels(odbs[0][1], "X_BEAM_2_END_NEG") or []
pos_labels = find_node_labels(odbs[0][1], "X_BEAM_2_END_POS") or []
if not neg_labels or not pos_labels:
    print("WARNING: X_BEAM_2_END_NEG/POS node sets not found; "
          "skipping central-clamp trace.")
else:
    central_rows = []
    for job_name, step in all_steps:
        u1_neg, u1_pos = {}, {}
        rf1_neg, rf1_pos = {}, {}
        t_local = None
        for name, region in step.historyRegions.items():
            if not name.startswith("Node"):
                continue
            label = int(name.split(".")[-1])
            if label not in neg_labels and label not in pos_labels:
                continue
            outs = region.historyOutputs
            if "U1" in outs.keys():
                data = outs["U1"].data
                if t_local is None:
                    t_local = [t for t, _ in data]
                vals = [v for _, v in data]
                (u1_neg if label in neg_labels else u1_pos)[label] = vals
            if "RF1" in outs.keys():
                vals = [v for _, v in outs["RF1"].data]
                (rf1_neg if label in neg_labels else rf1_pos)[label] = vals
        if t_local is None:
            print("  [no node history for %s/%s -- skipping central-clamp "
                  "trace for this phase]" % (job_name, step.name))
            continue
        n = len(t_local)
        for k in range(n):
            u_left = (sum(v[k] for v in u1_neg.values()) / len(u1_neg)
                      if u1_neg else float("nan"))
            u_right = (sum(v[k] for v in u1_pos.values()) / len(u1_pos)
                       if u1_pos else float("nan"))
            f_left = -sum(v[k] for v in rf1_neg.values()) if rf1_neg else 0.0
            f_right = -sum(v[k] for v in rf1_pos.values()) if rf1_pos else 0.0
            central_rows.append((t_local[k] + t0[id(step)], job_name,
                                  u_left, u_right, f_left, f_right))

    with open(base + "_central_clamp.csv", "w") as f:
        f.write("t_total,job,U1_left,U1_right,F_left,F_right\n")
        for r in central_rows:
            f.write("%.9e,%s,%.9e,%.9e,%.9e,%.9e\n" % r)
    print("wrote %s_central_clamp.csv  (%d rows across %d steps/%d jobs)"
          % (base, len(central_rows), len(all_steps), len(odbs)))

# --------------------------------------------- final deformed shape (field)
# Reuse the SAME step object already stored in all_steps/t0 (keyed by
# id()) rather than re-fetching via odb.steps[name] -- Abaqus's odbAccess
# API returns a fresh wrapper object on each such access, so a second
# fetch has a DIFFERENT id() than the one t0 was built with, breaking
# the id()-keyed lookup despite referring to "the same" step.
last_job_name, last_step = all_steps[-1]
last_odb = odbs[-1][1]
last_step_name = last_step.name
if len(last_step.frames) > 0:
    inst = last_odb.rootAssembly.instances[last_odb.rootAssembly.instances.keys()[0]]
    last = last_step.frames[-1]
    disp = last.fieldOutputs["U"]
    with open(base + "_final_shape.csv", "w") as f:
        f.write("label,x,y,z\n")
        coords = dict((n.label, n.coordinates) for n in inst.nodes)
        for v in disp.values:
            c = coords[v.nodeLabel]
            d = v.data
            f.write("%d,%.9e,%.9e,%.9e\n"
                    % (v.nodeLabel, c[0] + d[0], c[1] + d[1], c[2] + d[2]))
    print("wrote %s_final_shape.csv (job %s, step %s, t_total=%.4f)"
          % (base, last_job_name, last_step_name,
             t0[id(last_step)] + last.frameValue))

for _, odb in odbs:
    odb.close()
