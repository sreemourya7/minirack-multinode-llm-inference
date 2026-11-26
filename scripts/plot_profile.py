#!/usr/bin/env python3
"""Draws docs/part6-profile.svg from the numbers tests/check_part6.py saved.

    python3 tests/check_part6.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
    python3 scripts/plot_profile.py

Four stacked bars on one axis (milliseconds per token, one rank): prefill and
decode, each with the scalar and the AVX2 kernels, split by where the time
goes. A marker on the decode rows shows the memory roofline: the time needed
just to read the weights once.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Slots 1-3 of a validated categorical palette; text stays in the text inks.
SERIES = [("matmul_ms", "Matrix multiplies", "#257de0"),
          ("head_ms", "Vocabulary projection", "#f47230"),
          ("rest_ms", "Everything else", "#00b894")]
INK, MUTED, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "build", "logs", "part6_profile.csv"),
                    help="default: the file the check just wrote; the numbers behind the committed chart are in docs/data/")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "part6-profile.svg"))
    args = ap.parse_args()

    rows = []
    roofline = None
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            r["rest_ms"] = float(r["gelu_ms"]) + float(r["other_ms"])
            for k in ("matmul_ms", "head_ms", "total_ms"):
                r[k] = float(r[k])
            if r["roofline_ms"]:
                roofline = float(r["roofline_ms"])
            rows.append(r)

    W = 760
    left, right, top = 150, 70, 104
    bar_h, row_gap, group_gap = 20, 14, 30
    pw = W - left - right
    x_max = 20 * (int(max(r["total_ms"] for r in rows) / 20) + 1)

    def px(ms):
        return left + pw * ms / x_max

    # Vertical layout: rows grouped by stage.
    ys, y = [], top
    for i, r in enumerate(rows):
        if i and r["stage"] != rows[i - 1]["stage"]:
            y += group_gap - row_gap
        ys.append(y)
        y += bar_h + row_gap
    plot_bottom = y - row_gap + 10
    H = plot_bottom + 46

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'font-family="{FONT}">',
         "<title>Time per token before and after the AVX2 kernels, by where it goes</title>",
         "<desc>" + " ".join(f"{r['stage'].capitalize()} with {r['kernels']} kernels: {r['total_ms']:.1f} ms per token."
                             for r in rows) +
         (f" The memory roofline for decode is {roofline:.1f} ms." if roofline else "") + "</desc>",
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="24" y="32" font-size="17" font-weight="600" fill="{INK}">AVX2 made prefill about 6x faster; decode '
         f"ran into the memory roofline</text>",
         f'<text x="24" y="54" font-size="13" fill="{MUTED}">Time per token on one rank (GPT-2 small, 32-token '
         f"prompt, then 32 decoded tokens)</text>"]

    lx = 24
    for _, name, colour in SERIES:
        o.append(f'<rect x="{lx}" y="68" width="12" height="12" rx="3" fill="{colour}"/>')
        o.append(f'<text x="{lx + 18}" y="78" font-size="12" fill="{INK}">{name}</text>')
        lx += 36 + 6.6 * len(name)

    # Gridlines and x ticks.
    t = 0
    while t <= x_max:
        x = px(t)
        o.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{plot_bottom + 18}" font-size="12" fill="{MUTED}" text-anchor="middle">{t}</text>')
        t += 20
    o.append(f'<text x="{left + pw / 2}" y="{H - 8}" font-size="12" fill="{MUTED}" text-anchor="middle">'
             f"milliseconds per token</text>")

    # Bars: segments separated by a 2px gap in the surface colour; only the
    # outer end of the last segment is rounded.
    for r, y in zip(rows, ys):
        o.append(f'<text x="{left - 12}" y="{y + bar_h / 2 + 4}" font-size="13" fill="{INK}" text-anchor="end">'
                 f"{r['stage'].capitalize()}, {r['kernels']}</text>")
        x = left
        segments = [(key, name, colour, r[key]) for key, name, colour in SERIES if r[key] * pw / x_max >= 0.75]
        for j, (key, name, colour, ms) in enumerate(segments):
            w = ms * pw / x_max
            last = j == len(segments) - 1
            draw_w = max(w - (0 if last else 2), 0.75)
            if last and draw_w > 8:
                rr = 4
                o.append(f'<path d="M{x:.1f} {y} h{draw_w - rr:.1f} a{rr} {rr} 0 0 1 {rr} {rr} v{bar_h - 2 * rr} '
                         f'a{rr} {rr} 0 0 1 -{rr} {rr} h-{draw_w - rr:.1f} z" fill="{colour}">'
                         f"<title>{name}: {ms:.1f} ms</title></path>")
            else:
                o.append(f'<rect x="{x:.1f}" y="{y}" width="{draw_w:.1f}" height="{bar_h}" fill="{colour}">'
                         f"<title>{name}: {ms:.1f} ms</title></rect>")
            x += w
        o.append(f'<text x="{px(r["total_ms"]) + 8:.1f}" y="{y + bar_h / 2 + 4}" font-size="13" font-weight="600" '
                 f'fill="{INK}">{r["total_ms"]:.1f} ms</text>')

    # The roofline marker, on the decode rows only.
    if roofline:
        decode_ys = [y for r, y in zip(rows, ys) if r["stage"] == "decode"]
        x = px(roofline)
        y0, y1 = decode_ys[0] - 8, decode_ys[-1] + bar_h + 6
        o.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{INK}" stroke-width="1.5"/>')
        o.append(f'<text x="{x + 6:.1f}" y="{y0 - 6}" font-size="12" fill="{INK}">memory roofline, {roofline:.0f} ms: '
                 f"the time just to read the weights once</text>")

    o.append("</svg>")
    with open(args.out, "w") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
