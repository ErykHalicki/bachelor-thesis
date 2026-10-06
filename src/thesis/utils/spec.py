LAST = "last"
ACTION = "action"


def _parse_part(part):
    if isinstance(part, str):
        if part.strip() == LAST:
            return [LAST]
        if ".." in part:
            lo, hi = part.split("..")
            return list(range(int(lo), int(hi) + 1))
    return [int(part)]


def parse_index(index):
    """Index spec -> list of ints, plus the `"last"` sentinel for the episode-final frame.

    Accepts a list, or a compact string of comma-separated entries where each entry is an
    int, an inclusive range "a..b", or "last":
        "0..47"        -> [0, 1, ..., 47]
        "-12..-1"      -> [-12, -11, ..., -1]
        "-8..-2, 0"    -> [-8, -7, ..., -2, 0]
        "last"         -> ["last"]   (the episode's final frame, e.g. a goal image)
    A `"last"` entry is not a relative offset: the dataset resolves it to the last frame of
    the sample's episode, and at eval it is sourced from the env goal image.
    """
    parts = index.split(",") if isinstance(index, str) else index
    return [i for part in parts for i in _parse_part(part)]


def spec_fields(name, spec):
    """The batch fields a spec entry reads, always as a list.

    `from` is normally one field name (default: the entry name). A list instead means the
    entry reads several fields with one window and stitches them into a single input --
    two camera views side by side through one encoder pass, rather than a token stream
    each. Everything downstream of the stitch sees one stream.
    """
    field = spec.get("from", name)
    return [field] if isinstance(field, str) else [str(f) for f in field]


def raw_index(spec):
    """The raw rows a spec entry's batch field is loaded at, defaulting to its `index`.

    They differ when one index step of a stream is made of several raw rows: a V-JEPA
    tubelet is 2 frames, so `index` counts tubelets while `raw_index` lists the frames
    they are built from, `raw_steps_per_index` of them per index step, in order. It is
    also where a stream's raw sampling rate lives -- 10 frames at 5 fps out of a 30 fps
    recording is a `raw_index` strided by 6.

    Dataset backends load `raw_index`; the trunk's token layout and RoPE positions read
    `index`. With one raw row per index step (everything but true tubelets) they are the
    same list and neither side has to know the difference.
    """
    return spec.get("raw_index", spec["index"])


def split_index(index):
    """parse_index split into (sorted relative int offsets, has_last)."""
    steps = parse_index(index)
    has_last = LAST in steps
    relative = sorted(s for s in steps if s != LAST)
    return relative, has_last


def window_lengths(modality_spec):
    """`{key: {"index": ...}} -> {key: len(index)}`."""
    return {key: len(parse_index(spec["index"])) for key, spec in modality_spec.items()}


def flow_source_entries(predict_spec):
    """Pseudo spec entries for the raw windows that `source:` blocks of flow streams read
    (A2A seeding, see algorithms/README.md): `{"<stream>.source": {"from": ..., "index": ...}}`.

    Only field-mode sources (those naming a batch field via `from`/`index`) appear here;
    a stream-mode source (`stream:`) reuses an encoded conditioning stream and needs no
    rows of its own. Dataset backends fold these into their window unions exactly like
    real entries, so the seed rows arrive in the batch; the algorithm registers matching
    selectors. `"last"` is rejected: a seed is history, and the episode-final frame is not.
    """
    out = {}
    for name, spec in dict(predict_spec or {}).items():
        src = spec.get("source") if hasattr(spec, "get") else None
        if not src or src.get("stream"):
            continue
        if "index" not in src:
            raise ValueError(
                f"flow stream '{name}': a field-mode `source:` needs an `index` naming "
                f"the raw rows that seed it (or use `stream:` to reuse an encoded stream)"
            )
        if LAST in parse_index(src["index"]):
            raise ValueError(
                f"flow stream '{name}': a source seed is history; '{LAST}' is not allowed"
            )
        out[f"{name}.source"] = {
            "from": src.get("from") or spec.get("from", name),
            "index": src["index"],
        }
    return out


def action_entry(modality_spec):
    """`(name, field)` of the entry carrying the robot's actions, or None if there is none.

    Stream names are otherwise arbitrary, but a few things have to know which stream the
    actions are: how long a chunk is, which prediction a rollout executes, and which
    statistics unnormalize it back into the arm's units. An entry claims that with
    `role: action`; without one, an entry reading the `action` field claims it by
    convention, which is what every config did before the role existed.

    Two claimants is a build-time error rather than a first-past-the-post pick, since
    every caller assumes exactly one.
    """
    for name, entry in modality_spec.items():
        role = entry.get("role")
        if role is not None and role != ACTION:
            raise ValueError(
                f"entry '{name}' declares unknown role '{role}'; the only role is '{ACTION}'."
            )
    claimed = [n for n, e in modality_spec.items() if e.get("role") == ACTION]
    found = claimed or [n for n, e in modality_spec.items() if e.get("from", n) == ACTION]
    if len(found) > 1:
        raise ValueError(
            f"entries {sorted(found)} all carry actions; exactly one may. "
            f"Mark it with `role: {ACTION}`."
        )
    if not found:
        return None
    return found[0], modality_spec[found[0]].get("from", found[0])
