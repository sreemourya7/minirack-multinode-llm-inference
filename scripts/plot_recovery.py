#!/usr/bin/env python3
"""Draws docs/part5-recovery.svg from the numbers tests/check_part5.py saved.

    python3 tests/check_part5.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
    python3 scripts/plot_recovery.py

No plotting library needed: the chart is simple enough to write as SVG by hand.
Two lines on one linear axis: time to rebuild a session's cache on a
replacement rank, against the session's length, for the two strategies.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Colours: slots 1 and 2 of a validated categorical palette (colour-blind safe
# as a pair). Text is always in the text inks, never in a series colour.
BLUE, ORANGE = "#257de0", "#f47230"
INK, MUTED, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def fmt_time(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def nice_ceiling(v):
    for step in (1, 2, 5, 10, 20, 50):
        if v <= step * 6:
            return step * 6, step
    return v, v / 6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "build", "logs", "part5_recovery.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "part5-recovery.svg"))
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    xs = [int(r["tokens"]) for r in rows]
    replay = [float(r["replay_rebuild_ms"]) for r in rows]
    checkpoint = [float(r["checkpoint_rebuild_ms"]) for r in rows]

    W, H = 760, 420
    left, right, top, bottom = 64, 150, 96, 56  # the wide right margin holds the end labels
    pw, ph = W - left - right, H - top - bottom
    x_max = max(xs)
    y_max_s, y_step = nice_ceiling(max(replay) / 1000)

    def px(x):
        return left + pw * x / x_max

    def py(ms):
        return top + ph * (1 - ms / 1000 / y_max_s)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'font-family="{FONT}">',
         "<title>Time to rebuild a session after a rank fails, by session length</title>",
         f"<desc>Replay grows in step with session length, from {fmt_time(replay[0])} at {xs[0]} tokens to "
         f"{fmt_time(replay[-1])} at {xs[-1]} tokens. Checkpoint stays between {fmt_time(min(checkpoint))} and "
         f"{fmt_time(max(checkpoint))}.</desc>",
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{left}" y="32" font-size="17" font-weight="600" fill="{INK}">Rebuilding a lost rank\'s cache: '
         f"replay grows with the session, checkpoint does not</text>",
         f'<text x="{left}" y="54" font-size="13" fill="{MUTED}">Time to rebuild one session on a replacement rank '
         f"(2 ranks, GPT-2 small, 2-core machine, scalar code)</text>"]

    # Legend: always present for two series. A short line-key beside ink-coloured text.
    lx = left
    for name, colour in (("Replay: re-run the tokens", BLUE), ("Checkpoint: copy the saved cache back", ORANGE)):
        o.append(f'<line x1="{lx}" y1="76" x2="{lx + 18}" y2="76" stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        o.append(f'<text x="{lx + 24}" y="80" font-size="12" fill="{INK}">{name}</text>')
        lx += 44 + 6.4 * len(name)

    # Gridlines and y ticks: hairline, solid, recessive.
    t = 0
    while t <= y_max_s + 1e-9:
        y = py(t * 1000)
        o.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{t:g} s</text>')
        t += y_step
    for x in xs:
        o.append(f'<text x="{px(x):.1f}" y="{top + ph + 20}" font-size="12" fill="{MUTED}" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{left + pw / 2}" y="{H - 10}" font-size="12" fill="{MUTED}" text-anchor="middle">'
             f"tokens in the session when the rank failed</text>")

    # The two lines: 2px, round joins, a ringed dot on every measured point.
    for values, colour in ((replay, BLUE), (checkpoint, ORANGE)):
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, values))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
        for x, v in zip(xs, values):
            o.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="4" fill="{colour}" stroke="{SURFACE}" stroke-width="2">'
                     f"<title>{x} tokens: {fmt_time(v)}</title></circle>")

    # Direct labels at the line ends only, in text ink.
    ex = px(xs[-1]) + 12
    o.append(f'<text x="{ex:.1f}" y="{py(replay[-1]) - 2:.1f}" font-size="13" font-weight="600" fill="{INK}">Replay</text>')
    o.append(f'<text x="{ex:.1f}" y="{py(replay[-1]) + 15:.1f}" font-size="13" fill="{INK}">{fmt_time(replay[-1])}</text>')
    o.append(f'<text x="{ex:.1f}" y="{py(checkpoint[-1]) - 12:.1f}" font-size="13" font-weight="600" fill="{INK}">Checkpoint</text>')
    o.append(f'<text x="{ex:.1f}" y="{py(checkpoint[-1]) + 5:.1f}" font-size="13" fill="{INK}">{fmt_time(checkpoint[-1])}</text>')
    o.append("</svg>")

    with open(args.out, "w") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
