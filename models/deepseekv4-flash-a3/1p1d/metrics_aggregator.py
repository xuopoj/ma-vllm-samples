#!/usr/bin/env python3
"""Aggregate /metrics from every per-instance vLLM engine behind the PD proxy.

Why this exists: each vLLM instance exposes its own Prometheus /metrics on its
own API port (here: 4 prefillers on 7100..7103, 16 decoders on 7100..7115 =
20 endpoints). vLLM has no built-in cross-instance aggregator, and the PD proxy
(load_balance_proxy_server_example.py) only routes /v1/* + /healthcheck -- it
has NO /metrics. In a flat network you just point Prometheus at all 20 ports.

But in K8s you often can only reach the *proxy* Service externally (the engines
have no Service of their own). The proxy node, however, CAN reach every engine
pod (cluster-internal pod-to-pod). So run this on the proxy / group-0 node: it
scrapes each engine /metrics, prefixes every series with role/instance labels,
and re-exposes the union on one port -- Prometheus scrapes ONE endpoint (the
proxy Service's aggregator port) and still gets per-instance data.

run.sh starts it automatically alongside the proxy when the env var
AISHIPBOX_USE_METRIC_AGGREGATOR is set (port from AISHIPBOX_METRIC_AGGREGATOR_PORT,
default 9100). It is a SEPARATE script -- not a sidecar and not a patch to the
vendored proxy. The proxy is vendored from vllm-ascend verbatim and must stay
byte-identical; keeping the aggregation here leaves it untouched.

Endpoints are derived the same way run_proxy.sh derives them (from the rank
table's AISHIPBOX_ADDR_0/_1, exported by setup_rank_env.sh), so this stays in
sync. Override with --prefiller-host/--decoder-host etc. for manual runs.

Modes:
  serve  (default) -- run an HTTP server exposing /metrics (Prometheus target)
                      and /healthcheck. Self-contained, stdlib only.
  dump            -- scrape once, print the merged exposition to stdout, exit.

Usage:
  # automatic: set in the ModelArts service env, run.sh does the rest
  AISHIPBOX_USE_METRIC_AGGREGATOR=1 [AISHIPBOX_METRIC_AGGREGATOR_PORT=9100]

  # manual (after setup_rank_env.sh has run so AISHIPBOX_ADDR_0/_1 are set):
  python3 metrics_aggregator.py --port 9100
  python3 metrics_aggregator.py dump            # one-shot to stdout

  # explicit endpoints (no rank-table env):
  python3 metrics_aggregator.py \
      --prefiller-host 10.0.0.1 --n-prefill 4 \
      --decoder-host   10.0.0.2 --n-decode  16 \
      --engine-port 7100
"""

import argparse
import concurrent.futures
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_targets(args):
    """Return [(role, instance_label, url), ...] for every engine /metrics.

    instance_label is what gets added as instance="..." on every series, so a
    scraped series stays attributable to one engine after the merge.
    """
    prefill_host = args.prefiller_host or os.environ.get("AISHIPBOX_ADDR_0")
    decode_host = args.decoder_host or os.environ.get("AISHIPBOX_ADDR_1")
    if not prefill_host or not decode_host:
        sys.exit(
            "[metrics] need prefill/decode hosts: pass --prefiller-host/--decoder-host, "
            "or run where AISHIPBOX_ADDR_0/_1 are set (via setup_rank_env.sh)."
        )

    targets = []
    for i in range(args.n_prefill):
        port = args.engine_port + i
        targets.append(("prefill", f"{prefill_host}:{port}", f"http://{prefill_host}:{port}/metrics"))
    for i in range(args.n_decode):
        port = args.engine_port + i
        targets.append(("decode", f"{decode_host}:{port}", f"http://{decode_host}:{port}/metrics"))
    return targets


def relabel(body: str, role: str, instance: str) -> str:
    """Inject role="..." and instance="..." into each metric sample line.

    Prometheus text format: keep # HELP/# TYPE lines as-is; for sample lines,
    splice our labels into the existing {...} (or add one if absent). Without
    this every instance's `vllm:num_requests_running` would collide on scrape.
    """
    extra = f'role="{role}",instance="{instance}"'
    out = []
    for line in body.splitlines():
        if not line or line.startswith("#"):
            out.append(line)
            continue
        name, sep, rest = line.partition("{")
        if sep:  # already has labels: metric{a="b"} 1  -> metric{role=..,a="b"} 1
            out.append(f"{name}{{{extra},{rest}")
        else:  # bare: metric 1  -> metric{role=..} 1
            metric, _, value = line.partition(" ")
            out.append(f"{metric}{{{extra}}} {value}")
    return "\n".join(out) + "\n"


def scrape_one(target, timeout):
    role, instance, url = target
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return role, instance, relabel(body, role, instance), None
    except (urllib.error.URLError, OSError) as e:
        # One dead engine must not blank the whole scrape; just record it.
        return role, instance, "", f"{instance}: {e}"


def scrape_all(targets, timeout, max_workers=16):
    chunks, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for role, instance, text, err in ex.map(lambda t: scrape_one(t, timeout), targets):
            chunks.append(text)
            # vllm_proxy_scrape_up{role,instance} = 1 if it answered, else 0, so
            # you can alert on a missing engine even when its body is empty.
            up = 0 if err else 1
            chunks.append(f'vllm_proxy_scrape_up{{role="{role}",instance="{instance}"}} {up}\n')
            if err:
                errors.append(err)
    return "".join(chunks), errors


def make_handler(args, targets):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # quiet; suppress request logging
            pass

        def do_GET(self):
            if self.path == "/healthcheck":
                self._send(200, '{"status":"ok"}\n', "application/json")
                return
            if self.path.rstrip("/") not in ("/metrics", ""):
                self._send(404, "not found\n")
                return
            body, errors = scrape_all(targets, args.timeout)
            if errors:
                print(f"[metrics] {len(errors)}/{len(targets)} engines unreachable: "
                      + "; ".join(errors), file=sys.stderr)
            self._send(200, body, "text/plain; version=0.0.4")

        def _send(self, code, body, ctype="text/plain"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="serve", choices=["serve", "dump"])
    p.add_argument("--host", default="0.0.0.0", help="bind host for serve mode")
    p.add_argument("--port", type=int, default=9100, help="bind port for serve mode (Prometheus target)")
    p.add_argument("--engine-port", type=int, default=7100, help="first per-instance API port (matches --vllm-start-port)")
    p.add_argument("--n-prefill", type=int, default=4, help="number of prefill instances (rank 0)")
    p.add_argument("--n-decode", type=int, default=16, help="number of decode instances (rank 1)")
    p.add_argument("--prefiller-host", help="prefill node IP (default: AISHIPBOX_ADDR_0)")
    p.add_argument("--decoder-host", help="decode node IP (default: AISHIPBOX_ADDR_1)")
    p.add_argument("--timeout", type=float, default=5.0, help="per-engine scrape timeout (s)")
    args = p.parse_args()

    targets = build_targets(args)

    if args.mode == "dump":
        body, errors = scrape_all(targets, args.timeout)
        if errors:
            print(f"[metrics] {len(errors)}/{len(targets)} unreachable: " + "; ".join(errors), file=sys.stderr)
        sys.stdout.write(body)
        return

    print(f"[metrics] aggregating {len(targets)} engine /metrics "
          f"({args.n_prefill} prefill + {args.n_decode} decode) -> http://{args.host}:{args.port}/metrics",
          file=sys.stderr)
    ThreadingHTTPServer((args.host, args.port), make_handler(args, targets)).serve_forever()


if __name__ == "__main__":
    main()
