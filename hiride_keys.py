"""The cell-identity key, in ONE place.

A "cell" is one experimental condition: everything that must match before two
results are averaged together. Getting this wrong does not raise -- it silently
averages two different experiments and reports the mean as if it were one, which
is only visible as an odd `n` in the collate table.

That bug shipped four separate times in this campaign (`bits`, then
`frames`/`encoding`, then `head`, then `augment`/`test_fuse`), every time
because the key was written out longhand in both hiride_collate.py and
hiride_stats.py and only one copy was updated. Both now import from here, so a
new axis is added once.

RULE: any CLI flag that changes what the network sees or how it is trained MUST
appear in cond_key or arch_key. When adding a flag to hiride_train.py, add it
here in the same commit.
"""

# Every field that participates in cell identity, with the value that means
# "default, omit from the key". Anything not listed here is metadata.
AXES = dict(bits=16, frames=1, encoding="raw", head="gap", augment=0, test_fuse=1,
            erode=2, depth_slab_mm=None, init=None, eligibility="cues")


def arch_key(r):
    """Architecture plus the training-side choices that redefine the model."""
    arch = r["arch"]
    if arch == "convnext_tiny" and r.get("init") == "scratch":
        arch += "/scratch"
    if r.get("head", "gap") != "gap":
        arch += f"/{r['head']}"
    if r.get("augment", 0):
        arch += f"/aug{r['augment']}"
    if r.get("test_fuse", 1) > 1:
        arch += f"/tf{r['test_fuse']}"
    return arch


def cond_key(r):
    """Mask condition plus the input-side edits layered on top of it."""
    cond = r["condition"]
    if r.get("bits", 16) < 16:
        cond += f"@{r['bits']}b"
    if r.get("frames", 1) > 1:
        cond += f"/f{r['frames']}"
    enc = r.get("encoding", "raw")
    if enc == "normals":
        cond += "/nrm"
    elif enc == "depth_sil":
        cond += "/dsil"
    # erode only matters where an eroded mask is actually taken; the default is
    # 2 everywhere else and would add noise to every key.
    if r["condition"] == "interior_only" and r.get("erode", 2) != 2:
        cond += f"/e{r['erode']}"
    # eligibility changes WHICH FRAMES are scored, so a full_body cell is not a
    # seed of the ladder cell with the same name -- it is a different population.
    if r.get("eligibility", "cues") != "cues":
        cond += f"/{r['eligibility']}"
    return cond


def cell_key(r):
    """Full grouping key: results sharing this are seeds of one experiment."""
    return (r["policy"], r.get("guard"), r["modality"], arch_key(r),
            cond_key(r), bool(r.get("permuted", False)))


def describe(r):
    """Human-readable cell name, for figure labels and tables."""
    return f"{r['modality']}/{arch_key(r)}/{cond_key(r)}"
