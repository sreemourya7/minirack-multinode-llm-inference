#!/usr/bin/env python3
"""Type a prompt, watch the rack continue it (greedy decoding, KV cache).

Start a rack first, for example:   scripts/run_local_tp.sh weights/gpt2.mrkw 2

    python3 scripts/generate.py --hf-dir weights/gpt2 --prompt "Hello, my dog is cute" --tokens 30

Needs `pip install transformers` and GPT-2's tokenizer files (vocab.json,
merges.txt, tokenizer.json) in --hf-dir. The model is GPT-2 small with greedy
decoding, so expect plain and often repetitive text. With an MoE rack
(untrained router) expect nonsense.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mr_client import Client  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2's tokenizer files")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=30, help="how many new tokens to generate")
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(args.hf_dir, local_files_only=True)
    prompt_ids = tok.encode(args.prompt)

    with Client(args.host, args.port) as c:
        session = 1
        print(args.prompt, end="", flush=True)
        t0 = time.perf_counter()
        token, _ = c.prefill(session, prompt_ids, argmax=True)   # the whole prompt at once
        t_prefill = time.perf_counter() - t0
        out = [token]
        t0 = time.perf_counter()
        try:
            while len(out) < args.tokens:
                print(tok.decode(out[-1:]), end="", flush=True)
                token, _ = c.decode(session, token, argmax=True)  # one token at a time, using the cache
                out.append(token)
            print(tok.decode(out[-1:]), flush=True)
        finally:
            c.release(session)
        t_decode = time.perf_counter() - t0

    steps = max(len(out) - 1, 1)
    print(f"\nprefill: {len(prompt_ids)} tokens in {t_prefill * 1e3:.0f} ms   "
          f"decode: {t_decode / steps * 1e3:.1f} ms/token ({steps / t_decode:.1f} tokens/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
