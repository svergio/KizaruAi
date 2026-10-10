"""RL fine-tune MiniDeepSeek with Jeff rhyme LoRA as reward stick.

Phase 4 / A4: load SFT policy, sample candidate pairs from the same seed,
score with Jeff p_rhyme preference, GRPO (+ KL to frozen SFT ref).
Defaults to cuda when available (--device cpu hides GPU for mutex with SFT).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def _early_device_guard() -> None:
    if "--device" not in sys.argv:
        return
    i = sys.argv.index("--device")
    want = sys.argv[i + 1] if i + 1 < len(sys.argv) else "cuda"
    if want == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


_early_device_guard()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.text.checkpoint import load_model, save_model
from core.text.sample import _completed_lyric_lines_tokens, _sample_index
from core.text.tokenize import TOKEN_UNK, detokenize, encode_words

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = "checkpoints/deepseek_v2.pt"
DEFAULT_JEFF = "checkpoints/jef_rhyme_lora"
DEFAULT_OUT = "checkpoints/deepseek_rl_v1.pt"
DEFAULT_CORPUS = "data/corpus/kizaru_lyrics_ready.txt"
DEFAULT_STATUS = "var/deepseek_rl_status.json"
BIN_RHYME = 1


def parse_args():
    p = argparse.ArgumentParser(
        description="GRPO / REINFORCE RL for MiniDeepSeek with Jeff rhyme reward"
    )
    p.add_argument("--policy", default=DEFAULT_POLICY, help="SFT checkpoint (policy + ref)")
    p.add_argument("--jeff", default=DEFAULT_JEFF, help="Jeff LoRA adapter dir")
    p.add_argument("--jeff-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="Seed lines source")
    p.add_argument("--status", default=DEFAULT_STATUS)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--gen-len", type=int, default=48)
    p.add_argument("--max-lines", type=int, default=4)
    p.add_argument("--group-size", type=int, default=2, help="Candidates per seed (GRPO)")
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="MLflow metrics (+ verbose print) every N steps",
    )
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--jeff-max-len", type=int, default=256)
    p.add_argument("--seed-max-chars", type=int, default=120)
    p.add_argument("--algo", choices=("grpo", "reinforce"), default="grpo")
    p.add_argument(
        "--jeff-gpu-max-mem",
        default="6GiB",
        help="Jeff max_memory GPU budget (use 0GiB to keep Jeff on CPU only)",
    )
    p.add_argument("--jeff-cpu-max-mem", default="48GiB")
    p.add_argument("--jeff-offload", default="var/jef_rl_offload")
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_seed_pool(corpus: Path, max_chars: int, limit: int = 4000) -> list[str]:
    if not corpus.is_file():
        return ["Я на районе", "Слышь", "Ночь", "Деньги"]
    text = corpus.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pool = []
    step = max(1, len(lines) // limit)
    for i in range(0, len(lines), step):
        chunk = lines[i]
        if i + 1 < len(lines):
            chunk = chunk + "\n" + lines[i + 1]
        chunk = chunk[:max_chars].strip()
        if len(chunk) >= 3:
            pool.append(chunk)
        if len(pool) >= limit:
            break
    return pool or ["Я"]


def hard_gate(text: str, max_lines: int) -> float:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.15
    if len(lines) > max_lines + 2:
        return 0.4
    uniq = len(set(ln.lower() for ln in lines))
    if uniq < max(1, len(lines) - 1):
        return 0.35
    mean_len = float(np.mean([len(ln) for ln in lines]))
    if mean_len < 8 or mean_len > 120:
        return 0.5
    return 1.0


@torch.no_grad()
def sample_continuation(
    model,
    token_to_idx,
    idx_to_token,
    start_text: str,
    *,
    gen_len: int,
    temp: float,
    max_lines: int,
    device: torch.device,
):
    model.eval()
    ids = encode_words(start_text, token_to_idx)
    start_n = len(ids)
    start_lines = _completed_lyric_lines_tokens(ids, idx_to_token)
    block_size = model.block_size
    block = [token_to_idx[TOKEN_UNK]] if TOKEN_UNK in token_to_idx else None
    logprobs = []

    for _ in range(gen_len):
        ctx = torch.as_tensor(ids[-block_size:], dtype=torch.long, device=device).view(1, -1)
        logits = model(ctx)[0, -1]
        probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
        probs_np = probs.detach().cpu().numpy()
        idx = _sample_index(probs_np, len(token_to_idx), block_ids=block)
        logprobs.append(torch.log(probs[idx].clamp_min(1e-12)))
        ids.append(idx)
        if (
            max_lines is not None
            and _completed_lyric_lines_tokens(ids, idx_to_token) - start_lines >= max_lines
        ):
            break

    cont_ids = ids[start_n:]
    tokens = [
        idx_to_token[int(i)]
        for i in ids
        if idx_to_token.get(int(i)) != TOKEN_UNK
    ]
    text = detokenize(tokens).rstrip()
    sum_lp = torch.stack(logprobs).sum() if logprobs else torch.zeros((), device=device)
    return {
        "text": text,
        "cont_ids": cont_ids,
        "full_ids": ids,
        "sum_logprob": sum_lp,
        "n_tokens": len(cont_ids),
    }


def sequence_logprob(model, full_ids: list[int], cont_ids: list[int], device: torch.device):
    if not cont_ids:
        return torch.zeros((), device=device)
    start = len(full_ids) - len(cont_ids)
    block = model.block_size
    total = torch.zeros((), device=device)
    for t, tok in enumerate(cont_ids):
        pos = start + t
        ctx = full_ids[max(0, pos - block) : pos]
        if not ctx:
            continue
        x = torch.as_tensor(ctx, dtype=torch.long, device=device).view(1, -1)
        logits = model(x)[0, -1]
        total = total + F.log_softmax(logits, dim=-1)[int(tok)]
    return total


class JeffReward:
    def __init__(self, args, device: torch.device):
        from training.train_jef import build_model

        use_4bit = (args.device == "cuda") and torch.cuda.is_available()
        adapter = Path(args.jeff)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Jeff adapter missing: {adapter}")
        self.device = device
        self.max_len = args.jeff_max_len
        self.model, self.tok = build_model(
            args.jeff_model,
            use_4bit,
            lora_r=16,
            lora_alpha=32,
            device=device,
            adapter_dir=adapter,
            gpu_max_mem=args.jeff_gpu_max_mem if args.device == "cuda" else "0GiB",
            cpu_max_mem=args.jeff_cpu_max_mem,
            offload_folder=args.jeff_offload,
            for_train=False,
        )
        self.model.eval()

    @torch.inference_mode()
    def p_rhyme(self, text: str) -> float:
        enc = self.tok(
            text or " ",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        try:
            first = next(self.model.parameters())
            ids = ids.to(first.device)
            mask = mask.to(first.device)
        except StopIteration:
            pass
        out = self.model(input_ids=ids, attention_mask=mask, use_scheme=False)
        probs = torch.softmax(out.logits.float(), dim=-1)[0]
        return float(probs[BIN_RHYME].item())


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            print("cuda unavailable; falling back to cpu", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return torch.device("cpu")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    policy_path = Path(args.policy)
    out_path = Path(args.out)
    status_path = Path(args.status)
    if not policy_path.is_file():
        raise SystemExit(f"missing SFT policy: {policy_path}")

    write_status(
        status_path,
        {
            "state": "starting",
            "pid": os.getpid(),
            "device": str(device),
            "policy": str(policy_path),
            "jeff": args.jeff,
            "out": str(out_path),
            "started_at": _utc_now(),
            "steps": args.steps,
            "algo": args.algo,
        },
    )

    from training import mlflow_track

    mlflow_track.start_run(
        "kizaru-rl",
        run_name=f"rl_{out_path.stem}",
        params={
            "policy": str(policy_path),
            "jeff": args.jeff,
            "steps": args.steps,
            "algo": args.algo,
            "device": str(device),
            "out": str(out_path),
        },
        tags={"project": "kizaruai", "entrypoint": "train_rl_deepseek"},
    )

    print(f"load_policy {policy_path} device={device}", flush=True)
    policy, token_to_idx, idx_to_token, meta = load_model(policy_path, device=device)
    ref, _, _, _ = load_model(policy_path, device=device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    level = meta.get("level", "word")
    if level != "word":
        print(f"warn: checkpoint level={level!r}; word sampling still used", flush=True)

    print(f"load_jeff {args.jeff}", flush=True)
    jeff = JeffReward(args, device=torch.device("cpu") if device.type == "cpu" else device)

    seeds = load_seed_pool(Path(args.corpus), args.seed_max_chars)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)

    print(
        f"rl_start algo={args.algo} steps={args.steps} group={args.group_size} "
        f"kl={args.kl_coef} seeds={len(seeds)} vocab={len(token_to_idx)} "
        f"params={sum(p.numel() for p in policy.parameters())}",
        flush=True,
    )

    reward_ema = None
    best_mean_r = -1e9
    t0 = time.time()

    for step in range(1, args.steps + 1):
        policy.train()
        seed = seeds[int(np.random.randint(0, len(seeds)))]
        samples = []
        for _ in range(max(2, args.group_size)):
            samples.append(
                sample_continuation(
                    policy,
                    token_to_idx,
                    idx_to_token,
                    seed,
                    gen_len=args.gen_len,
                    temp=args.temp,
                    max_lines=args.max_lines,
                    device=device,
                )
            )

        rewards = []
        for s in samples:
            pr = jeff.p_rhyme(s["text"])
            gate = hard_gate(s["text"], args.max_lines)
            rewards.append(pr * gate)
            s["p_rhyme"] = pr
            s["gate"] = gate
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        if args.algo == "grpo":
            mean_r = rewards_t.mean()
            std_r = rewards_t.std(unbiased=False).clamp_min(1e-6)
            advantages = (rewards_t - mean_r) / std_r
        else:
            # REINFORCE with pair preference baseline (mean of group)
            advantages = rewards_t - rewards_t.mean()

        pg_terms = []
        kl_terms = []
        for i, s in enumerate(samples):
            # recompute logprob under current policy (grad) and frozen ref
            lp_pi = sequence_logprob(policy, s["full_ids"], s["cont_ids"], device)
            with torch.no_grad():
                lp_ref = sequence_logprob(ref, s["full_ids"], s["cont_ids"], device)
            n = max(1, s["n_tokens"])
            kl = (lp_pi.detach() - lp_ref) / n
            adv = advantages[i].detach()
            pg_terms.append(-adv * lp_pi)
            kl_terms.append(kl)

        loss_pg = torch.stack(pg_terms).mean()
        loss_kl = torch.stack(kl_terms).mean()
        loss = loss_pg + args.kl_coef * loss_kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        mean_r = float(rewards_t.mean().item())
        reward_ema = mean_r if reward_ema is None else 0.9 * reward_ema + 0.1 * mean_r
        prefer = int(rewards_t.argmax().item())

        log_every = max(1, int(args.log_every))
        do_log = step % log_every == 0 or step == 1
        if do_log:
            mlflow_track.log_metrics(
                {
                    "loss": float(loss.item()),
                    "pg": float(loss_pg.item()),
                    "kl": float(loss_kl.item()),
                    "R_mean": mean_r,
                    "R_ema": float(reward_ema),
                    "p_rhyme": float(sum(s["p_rhyme"] for s in samples) / max(len(samples), 1)),
                },
                step=step,
            )
            # verbose sample dump: every log_every, but skip flood when log_every==1
            # by printing full candidates every max(log_every, 5) except step 1
            if step == 1 or step % max(log_every, 5) == 0:
                print(
                    f"step={step}/{args.steps} loss={float(loss.item()):.4f} "
                    f"pg={float(loss_pg.item()):.4f} kl={float(loss_kl.item()):.4f} "
                    f"R_mean={mean_r:.3f} R_ema={reward_ema:.3f} "
                    f"p_rhyme={[round(s['p_rhyme'], 3) for s in samples]} "
                    f"prefer={prefer}",
                    flush=True,
                )
                print(f"seed={seed!r}", flush=True)
                print(f"A={samples[0]['text'][:240]!r}", flush=True)
                print(f"B={samples[1]['text'][:240]!r}", flush=True)
                print("-" * 40, flush=True)
            elif log_every == 1:
                print(
                    f"step={step}/{args.steps} loss={float(loss.item()):.4f} "
                    f"R_mean={mean_r:.3f} R_ema={reward_ema:.3f}",
                    flush=True,
                )

        payload = {
            "state": "running",
            "pid": os.getpid(),
            "device": str(device),
            "policy": str(policy_path),
            "jeff": args.jeff,
            "out": str(out_path),
            "started_at": _utc_now(),
            "step": step,
            "steps": args.steps,
            "algo": args.algo,
            "loss": float(loss.item()),
            "reward_mean": mean_r,
            "reward_ema": reward_ema,
            "kl": float(loss_kl.item()),
            "elapsed_sec": round(time.time() - t0, 1),
            "checked_at": _utc_now(),
        }
        write_status(status_path, payload)

        if mean_r > best_mean_r or step % args.save_every == 0 or step == args.steps:
            if mean_r > best_mean_r:
                best_mean_r = mean_r
            save_model(
                out_path,
                policy,
                token_to_idx,
                idx_to_token,
                epoch=step,
                mean_loss=float(loss.item()),
                level="word",
                rl_algo=args.algo,
                reward_ema=reward_ema,
                sft_policy=str(policy_path),
                jeff=args.jeff,
            )
            print(f"saved {out_path} step={step} R_mean={mean_r:.3f}", flush=True)

    write_status(
        status_path,
        {
            "state": "done",
            "pid": os.getpid(),
            "device": str(device),
            "out": str(out_path),
            "steps": args.steps,
            "reward_ema": reward_ema,
            "best_reward_mean": best_mean_r,
            "finished_at": _utc_now(),
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"done, model={out_path} reward_ema={reward_ema:.4f}", flush=True)
    mlflow_track.log_metrics({"R_ema": float(reward_ema), "best_R_mean": float(best_mean_r)})
    mlflow_track.end_run()


if __name__ == "__main__":
    main()
