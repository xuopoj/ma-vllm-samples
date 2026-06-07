#!/usr/bin/env python3
"""Simple smoke test for the A2 2P2D PD-disaggregated DeepSeek deployment.

Exercises the same P -> D KV handoff an external proxy would perform
(see examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
send_request_to_service / _handle_select_instance), but talks directly to one
Prefill engine and one Decode engine -- no proxy required:

  1. POST the prompt to the Prefill engine with `kv_transfer_params`
     seeded as {"do_remote_decode": true, "do_remote_prefill": false, ...}
     and max_tokens=1, so it computes the prompt's KV cache and returns
     `kv_transfer_params` describing where/how to fetch that cache remotely.
  2. POST the same prompt plus those `kv_transfer_params` to the Decode
     engine; it remote-reads the KV cache from the Prefill engine and
     continues generation -- its `choices[0].text` is the actual model
     output, proving the full prefill -> KV-transfer -> decode path works.

Usage:
  python3 smoke_test.py --prefill-url http://<p_ip>:7100 \\
                        --decode-url  http://<d_ip>:7100 \\
                        [--model deepseek_v4] [--prompt "..."] [--max-tokens 32]

Run it once per (Prefill, Decode) engine pair you want to validate, e.g.
prefill0<->decode0, prefill1<->decode1, and the cross pairs too.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill-url", required=True, help="Base URL of one Prefill engine, e.g. http://1.2.3.4:7100")
    parser.add_argument("--decode-url", required=True, help="Base URL of one Decode engine, e.g. http://1.2.3.5:7100")
    parser.add_argument("--model", default="deepseek_v4", help="Served model name (--served-model-name in run_*.sh)")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt to send")
    parser.add_argument("--max-tokens", type=int, default=32, help="max_tokens for the final decode response")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds")
    args = parser.parse_args()

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
    try:
        text = decode_resp["choices"][0]["text"]
    except (KeyError, IndexError):
        print(
            f"[smoke-test] FAIL: decode response has no choices[0].text:\n{json.dumps(decode_resp, indent=2)}",
            file=sys.stderr,
        )
        return 1

    print("[smoke-test] PASS — prefill -> KV transfer -> decode round trip succeeded")
    print(f"[smoke-test]    prompt: {args.prompt!r}")
    print(f"[smoke-test]    output: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
