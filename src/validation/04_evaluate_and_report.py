"""Build the validation report tables and cache all evaluation artifacts.

For each (condition x split):
  - Descriptive metrics (eligible-class Macro/Blank/Species/Head/Medium/Tail
    + WILDS `dataset.eval()` macro-F1 as a secondary column).
  - Per-class F1 cached under checkpoints/validation/results/per_class_f1/.

Paired bootstraps:
  - K=4+CE vs Frozen Linear CE (validates cycle 4's main effect).
  - K=4+cRT vs K=4+CE (validates cycle 4's cRT interaction; key cell is
    Tail x OOD test).
  Decomposed by Total/Head/Medium/Tail. Holm-Bonferroni applied across the
  two Total p-values within each test split. Per-segment p-values stay raw.

Three CSVs are written under checkpoints/validation/results/:
  - descriptive_metrics.csv
  - paired_bootstrap.csv
  - leaderboard_context.csv
"""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np
import torch
from _common import (
    CHECKPOINT_DIR,
    EVAL_SPLITS,
    MIN_CAM_IMAGES,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SEED,
    SPLIT_LABEL,
    compute_all_metrics,
    dataset_eval_macro_f1,
    get_wilds_dataset,
    holm_bonferroni,
    load_cached_features,
    load_segments,
    paired_cluster_bootstrap,
)
from sklearn.metrics import f1_score

# Conditions
CONDITIONS = [
    ("frozen_linear_ce", "Frozen Linear CE"),
    ("k4_ce", "K=4 + CE"),
    ("k4_crt", "K=4 + cRT"),
]

# Two paired comparisons (cycle 4 contract)
COMPARISONS = [
    ("k4_ce", "frozen_linear_ce"),  # validates main effect
    ("k4_crt", "k4_ce"),  # validates cRT interaction
]

# Test splits we run paired bootstraps on (val splits are used for model
# selection only, not headline reporting)
TEST_SPLITS = ["id_test", "test"]

# Published WILDS leaderboard numbers (macro-F1 on OOD test, dataset.eval()
# metric, all-class). Sources are cited inline.
LEADERBOARD = [
    {
        "method": "ERM (ResNet-50)",
        "backbone": "ResNet-50",
        "regime": "fully fine-tuned",
        "ood_test_macro_f1": 0.310,
        "source": "Koh et al., 2021 (WILDS) — leaderboard baseline",
    },
    {
        "method": "ABSGD",
        "backbone": "ResNet-50",
        "regime": "fully fine-tuned",
        "ood_test_macro_f1": 0.331,
        "source": "Qi et al., 2022 (ICML) — WILDS leaderboard",
    },
    {
        "method": "ERM (CLIP ViT-L)",
        "backbone": "CLIP ViT-L/14",
        "regime": "fully fine-tuned",
        "ood_test_macro_f1": 0.483,
        "source": "Wortsman et al., 2022 (Model Soups) — WILDS leaderboard",
    },
    {
        "method": "Model Soups",
        "backbone": "CLIP ViT-L/14",
        "regime": "fully fine-tuned (greedy soup)",
        "ood_test_macro_f1": 0.500,
        "source": "Wortsman et al., 2022 (ICML)",
    },
    {
        "method": "FLYP",
        "backbone": "CLIP ViT-L/14",
        "regime": "fully fine-tuned (contrastive)",
        "ood_test_macro_f1": 0.522,
        "source": "Goyal et al., 2023 (CVPR)",
    },
    {
        "method": "AutoFT",
        "backbone": "CLIP ViT-L/14",
        "regime": "fully fine-tuned",
        "ood_test_macro_f1": 0.543,
        "source": "Choi et al., 2024 (NeurIPS)",
    },
]


def load_predictions() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for cond_dir, _label in CONDITIONS:
        path = CHECKPOINT_DIR / cond_dir / "predictions.npz"
        npz = np.load(path)
        out[cond_dir] = {s: npz[f"{s}_preds"] for s in EVAL_SPLITS}
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "per_class_f1").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "bootstrap_samples").mkdir(parents=True, exist_ok=True)

    print("Loading cached features (for labels + metadata)…")
    cached = load_cached_features()
    segs = load_segments()

    rare = set(segs["rare_classes"])
    common = set(segs["common_classes"])
    frequent = set(segs["frequent_classes"])
    elig_per_split = {s: set(segs["eligible_per_split"][s]) for s in EVAL_SPLITS}

    print("Loading per-condition predictions…")
    preds = load_predictions()

    print("Loading WILDS dataset for dataset.eval()…")
    dataset = get_wilds_dataset()

    for split in EVAL_SPLITS:
        N = len(cached[f"{split}_labels"])
        for cond_dir, _ in CONDITIONS:
            assert len(preds[cond_dir][split]) == N, (
                f"length mismatch: {cond_dir}/{split} got "
                f"{len(preds[cond_dir][split])} expected {N}"
            )

    # 1) Descriptive table
    print("\nBuilding descriptive table…")
    desc_rows = []
    for cond_dir, label in CONDITIONS:
        for split in EVAL_SPLITS:
            tgt = cached[f"{split}_labels"].numpy()
            pr = preds[cond_dir][split]
            elig = elig_per_split[split]

            m = compute_all_metrics(pr, tgt, elig, rare, common, frequent)

            meta = cached[f"{split}_metadata"]
            de_f1 = dataset_eval_macro_f1(
                dataset,
                torch.from_numpy(pr).long(),
                cached[f"{split}_labels"].long(),
                meta.long(),
            )

            desc_rows.append(
                {
                    "condition": label,
                    "split": split,
                    "split_label": SPLIT_LABEL[split],
                    "macro_f1_eligible": round(m["Macro-F1"], 4),
                    "macro_f1_dataset_eval": round(de_f1, 4),
                    "blank_f1": round(m["Blank-F1"], 4),
                    "species_f1": round(m["Species-F1"], 4),
                    "head_f1": round(m["Head-F1"], 4),
                    "medium_f1": round(m["Medium-F1"], 4),
                    "tail_f1": round(m["Tail-F1"], 4),
                }
            )

            label_list = sorted(elig)
            per_cls = f1_score(
                tgt, pr, labels=label_list, average=None, zero_division=0
            )
            with open(
                RESULTS_DIR / "per_class_f1" / f"{cond_dir}__{split}.csv",
                "w",
                newline="",
            ) as f:
                w = csv.writer(f)
                w.writerow(["class_id", "f1", "support_in_split", "train_count"])
                for c, f1c in zip(label_list, per_cls):
                    sup = int((tgt == c).sum())
                    w.writerow(
                        [c, round(float(f1c), 4), sup, segs["train_class_counts"][c]]
                    )

    with open(RESULTS_DIR / "descriptive_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(desc_rows[0].keys()))
        w.writeheader()
        w.writerows(desc_rows)
    print(f"  wrote {RESULTS_DIR / 'descriptive_metrics.csv'}  ({len(desc_rows)} rows)")

    # 2) Paired bootstrap on test splits
    print(f"\nRunning paired cluster bootstraps (B={N_BOOTSTRAP})…")
    boot_rows = []

    for split in TEST_SPLITS:
        tgt = cached[f"{split}_labels"].numpy()
        meta = cached[f"{split}_metadata"].numpy()
        locs = meta[:, 0]
        elig = elig_per_split[split]

        cam_counts = Counter(locs)
        qual_cams = {c for c, n in cam_counts.items() if n >= MIN_CAM_IMAGES}
        qual_mask = np.isin(locs, list(qual_cams))
        n_qual_imgs = int(qual_mask.sum())
        print(f"  {split:8s} qualifying cameras={len(qual_cams)}  imgs={n_qual_imgs}")

        groups = {
            "Head": frequent & elig,
            "Medium": common & elig,
            "Tail": rare & elig,
        }

        comp_results = []
        for a_dir, b_dir in COMPARISONS:
            pr_a = preds[a_dir][split][qual_mask]
            pr_b = preds[b_dir][split][qual_mask]
            t = tgt[qual_mask]
            loc = locs[qual_mask]

            rng = np.random.default_rng(
                SEED + 100 + abs(hash((a_dir, b_dir, split))) % 10000
            )
            gap_overall, gap_by_group = paired_cluster_bootstrap(
                t,
                pr_a,
                pr_b,
                loc,
                eligible_classes=elig,
                n_boot=N_BOOTSTRAP,
                groups=groups,
                rng=rng,
            )
            comp_results.append((a_dir, b_dir, gap_overall, gap_by_group))

            np.save(
                RESULTS_DIR
                / "bootstrap_samples"
                / f"{a_dir}__vs__{b_dir}__{split}__total.npy",
                gap_overall["samples"],
            )
            for gname, g in gap_by_group.items():
                np.save(
                    RESULTS_DIR
                    / "bootstrap_samples"
                    / f"{a_dir}__vs__{b_dir}__{split}__{gname.lower()}.npy",
                    g["samples"],
                )

        total_p_raw = [r[2]["p_raw"] for r in comp_results]
        adjusted = holm_bonferroni(total_p_raw)

        for (a_dir, b_dir, gap_overall, gap_by_group), (adj_p, sig) in zip(
            comp_results, adjusted
        ):
            label_a = next(lab for d, lab in CONDITIONS if d == a_dir)
            label_b = next(lab for d, lab in CONDITIONS if d == b_dir)
            boot_rows.append(
                {
                    "comparison": f"{label_a} - {label_b}",
                    "split": split,
                    "split_label": SPLIT_LABEL[split],
                    "segment": "Total",
                    "delta_macro_f1": round(gap_overall["point"], 4),
                    "ci_low": round(float(gap_overall["ci"][0]), 4),
                    "ci_high": round(float(gap_overall["ci"][1]), 4),
                    "p_raw": round(gap_overall["p_raw"], 4),
                    "p_holm": round(adj_p, 4),
                    "significant_holm": bool(sig),
                }
            )
            for gname in ("Head", "Medium", "Tail"):
                g = gap_by_group[gname]
                boot_rows.append(
                    {
                        "comparison": f"{label_a} - {label_b}",
                        "split": split,
                        "split_label": SPLIT_LABEL[split],
                        "segment": gname,
                        "delta_macro_f1": round(g["point"], 4),
                        "ci_low": round(float(g["ci"][0]), 4),
                        "ci_high": round(float(g["ci"][1]), 4),
                        "p_raw": round(g["p_raw"], 4),
                        "p_holm": "",
                        "significant_holm": "",
                    }
                )

    with open(RESULTS_DIR / "paired_bootstrap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)
    print(f"  wrote {RESULTS_DIR / 'paired_bootstrap.csv'}  ({len(boot_rows)} rows)")

    # 3) Leaderboard context table
    print("\nBuilding leaderboard context table…")
    ours = next(
        r for r in desc_rows if r["condition"] == "K=4 + CE" and r["split"] == "test"
    )
    rows = [
        {
            "method": "Ours: K=4 + CE (DINOv2 ViT-B/14)",
            "backbone": "DINOv2 ViT-B/14",
            "regime": "partial fine-tune (last 4 blocks + final norm)",
            "ood_test_macro_f1": ours["macro_f1_dataset_eval"],
            "ood_test_macro_f1_eligible": ours["macro_f1_eligible"],
            "source": "this validation chapter (dataset.eval() metric)",
        }
    ]
    for r in LEADERBOARD:
        rows.append({**r, "ood_test_macro_f1_eligible": ""})

    with open(RESULTS_DIR / "leaderboard_context.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {RESULTS_DIR / 'leaderboard_context.csv'}  ({len(rows)} rows)")

    # Summary stdout
    print("\n" + "=" * 78)
    print("DESCRIPTIVE — eligible-class Macro-F1 (point estimates)")
    print("=" * 78)
    for cond_dir, label in CONDITIONS:
        line = f"  {label:<22s}"
        for split in EVAL_SPLITS:
            r = next(
                r for r in desc_rows if r["condition"] == label and r["split"] == split
            )
            line += f"  {split}={r['macro_f1_eligible']:.4f}"
        print(line)

    print("\nPAIRED BOOTSTRAP — Total cells (Holm-corrected)")
    print("=" * 78)
    for r in boot_rows:
        if r["segment"] != "Total":
            continue
        sig = " ✓" if r["significant_holm"] else ""
        print(
            f"  {r['comparison']:<35s} {r['split']:8s} "
            f"Δ={r['delta_macro_f1']:+.4f}  "
            f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
            f"p_holm={r['p_holm']:.4f}{sig}"
        )


if __name__ == "__main__":
    main()
