#!/usr/bin/env python3
"""Simple smoke test for the A2 2P2D PD-disaggregated DeepSeek deployment.

Two modes:

  --proxy-url URL
      The realistic end-to-end check: send a normal /v1/completions request
      to the load_balance_proxy_server_example.py proxy (e.g. the one started
      by run_proxy.sh) and verify it returns real generated text. The proxy
      itself performs the P -> KV-transfer -> D handoff, load-balancing across
      all configured prefill/decode engines -- this is what end users hit.

  --prefill-url URL --decode-url URL
      The diagnostic check: bypass the proxy and exercise one specific
      (Prefill, Decode) engine pair directly, performing the same handoff a
      proxy would (see examples/disaggregated_prefill_v1/
      load_balance_proxy_server_example.py: send_request_to_service /
      _handle_select_instance):
        1. POST the prompt to the Prefill engine with `kv_transfer_params`
           seeded as {"do_remote_decode": true, "do_remote_prefill": false,
           ...} and max_tokens=1, so it computes the prompt's KV cache and
           returns `kv_transfer_params` describing how to fetch it remotely.
        2. POST the same prompt plus those `kv_transfer_params` to the Decode
           engine; it remote-reads the KV cache and continues generation --
           `choices[0].text` is the real output. Useful for pinpointing which
           specific engine pair is broken when the proxy-level check fails
           (run it once per pair: prefill0<->decode0, prefill1<->decode1, and
           the cross pairs too).

Usage:
  python3 smoke_test.py --proxy-url   http://<proxy_ip>:8080 [--model deepseek_v4] ...
  python3 smoke_test.py --prefill-url http://<p_ip>:7100 \\
                        --decode-url  http://<d_ip>:7100  [--model deepseek_v4] ...
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"[smoke-test] HTTP {e.code} from {url}:\n{body}", file=sys.stderr)
        raise


def extract_text(resp: dict):
    try:
        return resp["choices"][0]["text"]
    except (KeyError, IndexError):
        return None


def run_via_proxy(args) -> int:
    url = args.proxy_url.rstrip("/") + "/v1/completions"
    req = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    print(f"[smoke-test] POST {url}  (prompt={args.prompt!r}, max_tokens={args.max_tokens})")
    print("[smoke-test] (the proxy performs prefill -> KV-transfer -> decode internally and load-balances)")
    resp = post_json(url, req, args.timeout)
    text = extract_text(resp)
    if text is None:
        print(f"[smoke-test] FAIL: proxy response has no choices[0].text:\n{json.dumps(resp, indent=2)}", file=sys.stderr)
        return 1

    print("[smoke-test] PASS — request round-tripped through the proxy (P -> KV transfer -> D)")
    print(f"[smoke-test]    prompt: {args.prompt!r}")
    print(f"[smoke-test]    output: {text!r}")
    return 0


def run_direct(args) -> int:
    prefill_url = args.prefill_url.rstrip("/") + "/v1/completions"
    decode_url = args.decode_url.rstrip("/") + "/v1/completions"

    prefill_req = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "stream": False,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    }
    print(f"[smoke-test] 1. POST {prefill_url}  (prompt={args.prompt!r}, max_tokens=1, do_remote_decode=true)")
    prefill_resp = post_json(prefill_url, prefill_req, args.timeout)
    kv_transfer_params = prefill_resp.get("kv_transfer_params")
    if not kv_transfer_params:
        print(
            f"[smoke-test] FAIL: prefill response has no kv_transfer_params:\n{json.dumps(prefill_resp, indent=2)}",
            file=sys.stderr,
        )
        return 1
    print(f"[smoke-test]    <- kv_transfer_params: {json.dumps(kv_transfer_params)}")

    decode_req = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "stream": False,
        "kv_transfer_params": kv_transfer_params,
    }
    print(f"[smoke-test] 2. POST {decode_url}  (max_tokens={args.max_tokens}, kv_transfer_params from step 1)")
    decode_resp = post_json(decode_url, decode_req, args.timeout)
    text = extract_text(decode_resp)
    if text is None:
        print(
            f"[smoke-test] FAIL: decode response has no choices[0].text:\n{json.dumps(decode_resp, indent=2)}",
            file=sys.stderr,
        )
        return 1

    print("[smoke-test] PASS — prefill -> KV transfer -> decode round trip succeeded")
    print(f"[smoke-test]    prompt: {args.prompt!r}")
    print(f"[smoke-test]    output: {text!r}")
    return 0


def main() -> int:
    examples = """\
Examples:
  # End-to-end through the proxy started by run.sh on a group-0 node:
  python3 smoke_test.py --proxy-url http://10.0.0.1:8080

  # Same, with a custom prompt and longer output:
  python3 smoke_test.py --proxy-url http://10.0.0.1:8080 \\
      --prompt "Write a haiku about snow" --max-tokens 64

  # Diagnose one engine pair directly (prefill0 -> decode0), bypassing the proxy:
  python3 smoke_test.py --prefill-url http://10.0.0.1:7100 --decode-url http://10.0.0.3:7100

  # Cross pair (prefill1 -> decode0) to isolate a broken link:
  python3 smoke_test.py --prefill-url http://10.0.0.2:7100 --decode-url http://10.0.0.3:7100
"""
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proxy-url", help="Base URL of the PD proxy, e.g. http://1.2.3.4:8080 (realistic end-to-end check)")
    parser.add_argument("--prefill-url", help="Base URL of one Prefill engine, e.g. http://1.2.3.4:7100 (diagnostic, pairs with --decode-url)")
    parser.add_argument("--decode-url", help="Base URL of one Decode engine, e.g. http://1.2.3.5:7100 (diagnostic, pairs with --prefill-url)")
    parser.add_argument("--model", default="deepseek_v4", help="Served model name (--served-model-name in run_*.sh)")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=32, help="max_tokens for the final response")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds")
    args = parser.parse_args()

    has_proxy = args.proxy_url is not None
    has_direct = args.prefill_url is not None or args.decode_url is not None
    if has_proxy and has_direct:
        parser.error("--proxy-url cannot be combined with --prefill-url/--decode-url; pick one mode")
    if has_proxy:
        return run_via_proxy(args)
    if args.prefill_url and args.decode_url:
        return run_direct(args)
    parser.error("specify either --proxy-url, or both --prefill-url and --decode-url")
    return 2  # unreachable, parser.error() exits


if __name__ == "__main__":
    sys.exit(main())
