#!/usr/bin/env python3
"""Draws docs/part5-recovery.svg: time to rebuild a lost rank's cache, by session length.

    python3 tests/check_part5.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
    python3 scripts/plot_recovery.py                      # from the run you just did
    python3 scripts/plot_recovery.py --csv docs/data/part5_recovery.csv   # the chart in the README

No plotting library needed: the chart is simple enough to write as SVG by hand.
Lines on one linear axis: replay and checkpoint, and optionally replay as
measured with the scalar kernels (MR_KERNELS=scalar), which shows how much a
faster prefill shortens recovery by replay.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Colours: slots 1-3 of a validated categorical palette (colour-blind safe in
# this order). Text is always in the text inks, never in a series colour.
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, MUTED, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def fmt_time(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def nice_axis(v):
    """Returns (axis maximum, tick step) with at most 6 steps, in the units of v."""
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50):
        if v <= step * 6:
            n = int(v / step) + 1
            return step * n, step
    return v, v / 6


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["tokens"]) for r in rows], [float(r["replay_rebuild_ms"]) for r in rows],
            [float(r["checkpoint_rebuild_ms"]) for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "build", "logs", "part5_recovery.csv"))
    ap.add_argument("--scalar-csv", help="optional: the same measurement made with MR_KERNELS=scalar")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "part5-recovery.svg"))
    args = ap.parse_args()

    xs, replay, checkpoint = read(args.csv)
    series = []  # (legend name, end label, colour, values), in fixed slot order
    if args.scalar_csv:
        xs_s, replay_s, _ = read(args.scalar_csv)
        if xs_s != xs:
            raise SystemExit("the two files must cover the same session lengths")
        series.append(("Replay, scalar kernels", "Replay, scalar", BLUE, replay_s))
        series.append(("Replay, AVX2 kernels", "Replay, AVX2", ORANGE, replay))
        series.append(("Checkpoint: copy the saved cache back", "Checkpoint", AQUA, checkpoint))
    else:
        series.append(("Replay: re-run the tokens", "Replay", BLUE, replay))
        series.append(("Checkpoint: copy the saved cache back", "Checkpoint", ORANGE, checkpoint))

    W, H = 760, 420
    left, right, top, bottom = 64, 176, 96, 56  # the wide right margin holds the end labels
    pw, ph = W - left - right, H - top - bottom
    x_max = max(xs)
    y_max_s, y_step = nice_axis(max(max(v) for *_, v in series) / 1000)

    def px(x):
        return left + pw * x / x_max

    def py(ms):
        return top + ph * (1 - ms / 1000 / y_max_s)

    summary = " ".join(f"{name}: {fmt_time(v[0])} at {xs[0]} tokens, {fmt_time(v[-1])} at {xs[-1]}." for name, _, _, v in series)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'font-family="{FONT}">',
         "<title>Time to rebuild a session after a rank fails, by session length</title>",
         f"<desc>{summary}</desc>",
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{left}" y="32" font-size="17" font-weight="600" fill="{INK}">Rebuilding a lost rank\'s cache: '
         f"replay grows with the session, checkpoint does not</text>",
         f'<text x="{left}" y="54" font-size="13" fill="{MUTED}">Time to rebuild one session on a replacement rank '
         f"(2 ranks, GPT-2 small)</text>"]

    # Legend: always present for two or more series. A short line-key beside ink-coloured text.
    lx = left
    for name, _, colour, _ in series:
        o.append(f'<line x1="{lx}" y1="76" x2="{lx + 18}" y2="76" stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<text x="{lx + 24}" y="80" font-size="12" fill="{INK}">{name}</text>')
        lx += 44 + 6.4 * len(name)

    # Gridlines and y ticks: hairline, solid, recessive.
    k = 0
    while k * y_step <= y_max_s + 1e-9:
        t = k * y_step
        y = py(t * 1000)
        o.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{t:g} s</text>')
        k += 1
    for x in xs:
        o.append(f'<text x="{px(x):.1f}" y="{top + ph + 20}" font-size="12" fill="{MUTED}" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{left + pw / 2}" y="{H - 10}" font-size="12" fill="{MUTED}" text-anchor="middle">'
             f"tokens in the session when the rank failed</text>")

    # The lines: 2px, round joins, a ringed dot on every measured point.
    for name, _, colour, values in series:
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, values))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
        for x, v in zip(xs, values):
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="4" fill="{colour}" stroke="{SURFACE}" stroke-width="2">'
                     f"<title>{name}, {x} tokens: {fmt_time(v)}</title></circle>")

    # Direct labels at the line ends only, in text ink: name in bold, then the value.
    # Labels that would sit on top of each other are pushed apart (lowest first).
    ex = px(xs[-1]) + 12
    spots = sorted(((py(v[-1]) + 4, label, v[-1]) for _, label, _, v in series), reverse=True)
    last_y = None
    for y, label, value in spots:
        if last_y is not None and last_y - y < 18:
            y = last_y - 18
        o.append(f'<text x="{ex:.1f}" y="{y:.1f}" font-size="13" fill="{INK}"><tspan font-weight="600">{label}</tspan> '
                 f"{fmt_time(value)}</text>")
        last_y = y
    o.append("</svg>")

    with open(args.out, "w") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
