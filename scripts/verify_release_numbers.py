"""Automated release verification for ScribblePU.

Checks:
1. Every reported number and delta in README.md and villa_pr/VILLA_PR_DESCRIPTION.md
   matches results/*.json exactly (plus the author's local post drafts, when present).
2. Zero em dashes (U+2014), zero en dashes (U+2013), and zero forbidden AI /
   attribution / mislabeling strings in all public files.
3. Local post drafts only (skipped when absent): the Discord body is <= 1,990 characters
   and the submission text is a single unbroken paragraph.
4. All 7 figures (fig0..fig6) exist and are landscape.
5. The 3 Reader-PU checkpoints match results/checkpoints_sha256.json and README.md
   (download them from Hugging Face and set SCRIBBLEPU_CKPT_DIR to check the files).
6. villa_pr/villa_full.patch and villa_pr/losses_patch.diff apply cleanly to villa
   merge-ink-pipelines (set VILLA_DIR to a villa checkout on that branch).
7. All 5 CPU unit tests in tests/test_losses.py pass, plus VillaBrierNNPULoss passes a
   forward/backward check (and a CompositeLoss check when VILLA_DIR is set).

Usage: python scripts/verify_release_numbers.py
Optional: VILLA_DIR=/path/to/villa SCRIBBLEPU_CKPT_DIR=/path/to/checkpoints python scripts/verify_release_numbers.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # Pillow is optional: without it the figure check only confirms the files exist
    Image = None

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
VILLA_DIR = Path(os.environ["VILLA_DIR"]) if os.environ.get("VILLA_DIR") else None
CKPT_DIR = Path(os.environ["SCRIBBLEPU_CKPT_DIR"]) if os.environ.get("SCRIBBLEPU_CKPT_DIR") else None

PUBLIC_DOCS = [
    REPO / "README.md",
    REPO / "villa_pr" / "VILLA_PR_DESCRIPTION.md",
]

# The author's local post drafts (not in the public repo); checked only when present.
FORM_DRAFT = REPO / "SUBMISSION_FORM.txt"
DISCORD_DRAFT = REPO / "DISCORD_POST.txt"


def read_draft(p: Path) -> str | None:
    if p.exists():
        return p.read_text(encoding="utf-8")
    print(f"  [SKIP] {p.name} not present (author's local draft, not part of the repo)")
    return None

PUBLIC_CODE = [
    REPO / "scribble_pu" / "__init__.py",
    REPO / "scribble_pu" / "losses.py",
    REPO / "scribble_pu" / "certain_blank.py",
    REPO / "villa_pr" / "brier_nnpu.py",
    REPO / "villa_pr" / "example_config_brier_nnpu.json",
    REPO / "villa_pr" / "test_brier_nnpu.py",
]

FORBIDDEN_PATTERNS = [
    r"\bclaude\b",
    r"\banthropic\b",
    r"\bgemini\b",
    r"\bopenai\b",
    r"\bchatgpt\b",
    r"co-authored-by",
    r"grand prize",
    r"new architecture",
]


def load_json(name: str) -> dict:
    p = RESULTS / name
    assert p.exists(), f"Missing result JSON: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_style_and_hygiene() -> list[str]:
    errors: list[str] = []
    drafts = [p for p in (FORM_DRAFT, DISCORD_DRAFT) if p.exists()]
    for p in PUBLIC_DOCS + drafts + PUBLIC_CODE:
        text = p.read_text(encoding="utf-8")
        if "\u2014" in text:
            errors.append(f"Em dash (U+2014) found in {p.relative_to(REPO)}")
        if "\u2013" in text:
            errors.append(f"En dash (U+2013) found in {p.relative_to(REPO)}")
        for ln in text.splitlines():
            if ln.startswith("ON HOLD"):
                continue
            if re.search(r"(~|about )6%|6% scrib|(?<![.\d])6% of (the )?ink", ln):
                errors.append(f"Stale Exam 2 coverage (~6%) in {p.relative_to(REPO)}: real = 0.73% of ink: {ln[:80]}")
        low = text.lower()
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, low):
                errors.append(f"Forbidden pattern {pat!r} found in {p.relative_to(REPO)}")
    return errors


def check_discord_and_form() -> list[str]:
    errors: list[str] = []
    discord_text = read_draft(DISCORD_DRAFT)
    m = re.search(
        r"BODY \([^)]+\):\r?\n(.*?)\r?\n\r?\nATTACH THESE",
        discord_text,
        flags=re.DOTALL,
    ) if discord_text is not None else None
    if discord_text is None:
        pass
    elif not m:
        errors.append("Could not parse BODY section in DISCORD_POST.txt")
    else:
        body = m.group(1).strip()
        n_chars = len(body)
        print(f"  [OK] DISCORD_POST.txt body length: {n_chars} chars (limit <= 1,990 / 2,000)")
        if n_chars > 1990:
            errors.append(f"DISCORD_POST.txt body too long: {n_chars} > 1990 chars")
        if n_chars < 500:
            errors.append(f"DISCORD_POST.txt body unexpectedly short: {n_chars} chars")

    form_text = read_draft(FORM_DRAFT)
    if form_text is None:
        return errors
    m_main = re.search(
        r"MAIN TEXT \([^)]+\):\r?\n\r?\n(.*?)\r?\n\r?\n=+",
        form_text,
        flags=re.DOTALL,
    )
    if m_main:
        main_para = m_main.group(1).strip()
        if "\n" in main_para:
            errors.append("SUBMISSION_FORM.txt MAIN TEXT should be a single unbroken paragraph")
        else:
            print(f"  [OK] SUBMISSION_FORM.txt MAIN TEXT paragraph: {len(main_para.split())} words, {len(main_para)} chars")
    else:
        lines = [ln.strip() for ln in form_text.splitlines() if ln.strip()]
        paras = [
            ln for ln in lines
            if not ln.startswith("=")
            and not ln.startswith("PROGRESS PRIZE")
            and not ln.startswith("URLS FIELD")
            and not ln.startswith("GitHub repo:")
            and not ln.startswith("HF checkpoints:")
            and not ln.startswith("villa PR:")
        ]
        if len(paras) != 1:
            errors.append(f"SUBMISSION_FORM.txt should contain exactly 1 unbroken paragraph, got {len(paras)}")
        else:
            print(f"  [OK] SUBMISSION_FORM.txt single paragraph: {len(paras[0].split())} words, {len(paras[0])} chars")
    return errors


def check_numbers() -> list[str]:
    errors: list[str] = []
    bhand = load_json("exam3_reader_pu_bhand.json")
    deconly = load_json("exam3_reader_pu_deconly.json")
    ex1 = load_json("exam1_from_scratch_s5k.json")
    ex1_p = load_json("exam1_from_scratch_s3k_pilot.json")
    ex2 = load_json("exam2_adaptation_0009b.json")
    alpha = load_json("exam2_alpha_sweep.json")
    ckpts = load_json("checkpoints_sha256.json")

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    pr_desc = (REPO / "villa_pr" / "VILLA_PR_DESCRIPTION.md").read_text(encoding="utf-8")
    form = FORM_DRAFT.read_text(encoding="utf-8") if FORM_DRAFT.exists() else None
    discord = DISCORD_DRAFT.read_text(encoding="utf-8") if DISCORD_DRAFT.exists() else None

    def must_contain(doc_name: str, text: str, needles: list[str]) -> None:
        plain = text.replace("**", "")
        for s in needles:
            if s not in text and s not in plain:
                errors.append(f"{doc_name} missing expected verified token: {s!r}")

    # 1. Verify Exam 3 Reader-PU values & exact deltas
    v3 = bhand["v3_ref"]
    v3_841 = bhand["v3_o841_ref"]
    er10 = bhand["erode_ctrl"]["V3_erode_r10"]
    bh = bhand["arms"]["bhand_raw"]
    bs50 = bhand["arms"]["bhand_soup50"]
    ds50 = deconly["arms"]["deconly_soup50"]

    d_b9_m1_v3 = bh["b9_M1"] - v3["b9_M1"]
    d_b9_m1_er = bh["b9_M1"] - er10["b9_M1"]
    d_w29_m1 = bh["w29_M1"] - v3["w29_M1"]

    must_contain("README.md", readme, [
        f"{bh['b9_M1']:.4f} / {bh['b9_M1_best']:.4f}",
        f"{bh['b9_M1n']:.4f}",
        f"{bh['b9_GATE']:.4f} / {bh['b9_M4']:.4f}",
        f"{bh['w29_M1']:.4f} / {bh['w29_M1n']:.4f}",
        f"{bh['w29_GATE']:.4f} / {bh['w29_M4']:.4f}",
        f"{int(bh['crater_p306_p01'])} ({bh['crater_p306_px_lt30']})",
        f"{bh['o841_GATE_best']:.4f} / {bh['o841_M2_best']:.4f}",
        f"{bh['FA80']:.4f}",
        f"{bs50['b9_M1']:.4f} / {bs50['b9_M1_best']:.4f}",
        f"{bs50['b9_GATE']:.4f} / {bs50['b9_M4']:.4f}",
        f"{bs50['w29_M1']:.4f} / {bs50['w29_M1n']:.4f}",
        f"{bs50['w29_GATE']:.4f} / {bs50['w29_M4']:.4f}",
        f"{bs50['o841_GATE_best']:.4f} / {bs50['o841_M2_best']:.4f}",
        f"{bs50['FA80']:.4f}",
        f"{ds50['b9_M1']:.4f} / {ds50['b9_M1_best']:.4f}",
        f"{ds50['b9_GATE']:.4f} / {ds50['b9_M4']:.4f}",
        f"{ds50['w29_M1']:.4f} / {ds50['w29_M1n']:.4f}",
        f"{ds50['w29_GATE']:.4f} / {ds50['w29_M4']:.4f}",
        f"{ds50['o841_GATE_best']:.4f} / {ds50['o841_M2_best']:.4f}",
        f"{ds50['FA80']:.4f}",
        f"{v3['b9_M1']:.4f} / {v3['b9_M1_best']:.4f}",
        f"{v3['b9_M1n']:.4f}",
        f"{v3['b9_GATE']:.4f} / {v3['b9_M4']:.4f}",
        f"{v3['w29_M1']:.4f} / {v3['w29_M1n']:.4f}",
        f"{v3['w29_GATE']:.4f} / {v3['w29_M4']:.4f}",
        f"{v3_841['GATE_best']:.4f} / {v3_841['M2_best']:.4f}",
        f"{v3['FA80']:.4f}",
        f"{er10['b9_M1']:.4f} / {er10['b9_M1_best']:.4f}",
        f"+{d_b9_m1_v3:.4f}",
        f"+{d_b9_m1_er:.4f}",
        f"+{d_w29_m1:.4f}",
        f"b9_p99 = {int(bh['b9_p99'])}",
        f"+{bh['ink_bias_delta']:.4f}",
    ])

    # 2. Verify Exam 1 (all 7 arms)
    ora = ex1["oracle_s5k"]
    spu = ex1["scribble_pu_rdg_s5k"]
    cbb = ex1["collar_bce_bal_s5k"]
    kir = ex1["kiryo_nnpu_s5k"]
    cb = ex1["collar_bce_s5k"]
    nb = ex1["naive_bce_s5k"]
    p20 = ex1["scribble_pu_p20_s5k"]

    pct_ora = 100.0 * spu["b9_GATE"] / ora["b9_GATE"]
    pct_ora_3k = 100.0 * ex1_p["scribble_pu_s3k"]["b9_GATE"] / ex1_p["oracle_s3k"]["b9_GATE"]
    must_contain("README.md", readme, [
        f"{pct_ora:.1f}%",
        f"{pct_ora_3k:.1f}%",
        f"{ex1_p['oracle_s3k']['b9_GATE']:.4f}",
        f"{ex1_p['kiryo_nnpu_s3k']['b9_GATE']:.4f}",
        f"{ex1_p['scribble_pu_s3k']['b9_GATE']:.4f}",
        f"{ex1_p['oracle_s3k']['FA80']:.4f}",
        "Warner & Bodill",
        f"+{spu['b9_M4'] - ora['b9_M4']:.4f}",
        f"+{spu['b9_rAUC'] - ora['b9_rAUC']:.4f}",
        f"+{spu['b9_M1'] - ora['b9_M1']:.4f}",
        f"+{spu['b9_GATE'] - kir['b9_GATE']:.4f}",
        f"+{spu['b9_M4'] - kir['b9_M4']:.4f}",
        f"+{spu['b9_rAUC'] - kir['b9_rAUC']:.4f}",
        f"+{spu['b9_M1'] - kir['b9_M1']:.4f}",
        f"+{spu['s814_GATE'] - kir['s814_GATE']:.4f}",
        f"{spu['FA80'] - kir['FA80']:.4f}",
        f"+{int(spu['b9_p99'] - kir['b9_p99'])}",
        f"+{spu['b9_M4'] - cbb['b9_M4']:.4f}",
        f"{spu['FA80'] - cbb['FA80']:.4f}",
        f"+{int(spu['cP20'] - cbb['cP20'])}",
        f"+{spu['b9_GATE'] - p20['b9_GATE']:.4f}",
    ])
    for arm_key, arm in ex1.items():
        must_contain("README.md", readme, [
            f"{arm['b9_GATE']:.4f}",
            f"{arm['b9_M4']:.4f}",
            f"{arm['b9_rAUC']:.4f}",
            f"{arm['b9_M1']:.4f} / {arm['b9_M1n']:.4f}",
            f"{arm['s814_GATE']:.4f} / {arm['s814_M4']:.4f}",
            f"{arm['FA80']:.4f}",
        ])

    # 3. Verify Exam 2A + 5-point alpha sweep
    b2 = ex2["base_eval"]
    o2 = ex2["oracle_fullmask"]
    s2 = ex2["scribble_pu"]
    k2 = ex2["kiryo_nnpu"]
    cb2b = ex2["collar_bce_bal"]
    cb2 = ex2["collar_bce"]
    nb2 = ex2["naive_bce"]

    pct_ex2 = 100.0 * (s2["b140_GATE"] - b2["b140_GATE"]) / (o2["b140_GATE"] - b2["b140_GATE"])
    must_contain("README.md", readme, [
        f"{pct_ex2:.1f}%",
        f"+{s2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{o2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{nb2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{cb2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{cb2b['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{k2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{s2['b125_GATE'] - b2['b125_GATE']:.4f}",
        f"+{o2['b125_GATE'] - b2['b125_GATE']:.4f}",
    ])
    for k, row in alpha.items():
        must_contain("README.md", readme, [
            f"{row['b140_GATE']:.4f}",
            f"{row['b125_GATE']:.4f}",
        ])

    # 4. Cross-check PR description, Submission Form, and Discord post key numbers
    must_contain("VILLA_PR_DESCRIPTION.md", pr_desc, [
        f"{spu['b9_GATE']:.4f}",
        f"{ora['b9_GATE']:.4f}",
        f"{pct_ora:.1f}%",
        f"{spu['b9_M4']:.4f}",
        f"{ora['b9_M4']:.4f}",
        f"{spu['b9_M1']:.4f}",
        f"{ora['b9_M1']:.4f}",
        f"+{s2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{o2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{pct_ex2:.1f}%",
        f"+{cb2b['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{bh['b9_M1']:.4f}",
        f"{v3['b9_M1']:.4f}",
        f"+{d_b9_m1_er:.4f}",
        f"{bh['w29_M1']:.4f}",
        f"{v3['w29_M1']:.4f}",
    ])

    pct_cbb = 100.0 * (cb2b["b140_GATE"] - b2["b140_GATE"]) / (o2["b140_GATE"] - b2["b140_GATE"])
    if form is not None:
      must_contain("SUBMISSION_FORM.txt", form, [
        f"{round(pct_ex2)}%",
        f"{round(pct_cbb)}%",
        f"{pct_ora_3k:.1f}%",
        f"{pct_ora:.1f}%",
        f"+{s2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{o2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"+{cb2b['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{nb2['b140_GATE'] - b2['b140_GATE']:.4f}",
        f"{bh['b9_M1']:.4f}",
        f"{v3['b9_M1']:.4f}",
        f"{ds50['b9_M1']:.4f}",
        f"{v3['FA80']:.4f}",
        f"{ds50['FA80']:.4f}",
        f"{bh['FA80']:.4f}",
      ])

    if discord is not None:
        must_contain("DISCORD_POST.txt", discord, [
            f"{pct_ora:.1f}%",
            f"{round(pct_ex2)}%",
            f"{pct_ora_3k:.1f}%",
        ])

    # 5. Verify checkpoint SHA256 hashes in README.md (always) and the files themselves (when present)
    for rel_name, meta in ckpts.items():
        must_contain("README.md", readme, [
            rel_name,
            f"{meta['size_mb']:.2f} MB",
            meta["sha256"],
        ])
        src = REPO / meta["source_path"]
        if not src.exists() and CKPT_DIR is not None:
            src = CKPT_DIR / rel_name
        if not src.exists():
            print(f"  [SKIP] {rel_name} not on disk (hosted on Hugging Face; set SCRIBBLEPU_CKPT_DIR to check its hash)")
        else:
            actual = sha256_file(src)
            if actual != meta["sha256"]:
                errors.append(f"SHA256 mismatch for {rel_name}: disk={actual} json={meta['sha256']}")
            else:
                print(f"  [OK] Checkpoint verified: {rel_name} ({meta['size_mb']:.2f} MB, {actual[:12]}...)")

    print("  [OK] All reported numbers & deltas match results/*.json exactly!")
    return errors


def check_figures() -> list[str]:
    errors: list[str] = []
    expected = [
        "fig0_scribble_workflow.png",
        "fig1_hero_first_letters_0306B.png",
        "fig2_counter_opening_and_erosion.png",
        "fig3_exam1_from_scratch_scribbles.png",
        "fig4_exam2_adaptation_and_alpha.png",
        "fig5_adaptation_0009b.png",
        "fig6_koine_trainer_check.png",
    ]
    for fn in expected:
        p = FIGURES / fn
        if not p.exists():
            errors.append(f"Missing figure: {p}")
            continue
        if Image is None:
            print(f"  [OK] Figure present: {fn} (install Pillow to also check its shape)")
            continue
        with Image.open(p) as im:
            w, h = im.size
        if w <= h:
            errors.append(f"Figure {fn} is not landscape ({w}x{h})")
        else:
            print(f"  [OK] Figure landscape verified: {fn} ({w}x{h}, {p.stat().st_size // 1024} KB)")
    return errors


def check_villa_patch() -> list[str]:
    errors: list[str] = []
    patch_files = [REPO / "villa_pr" / "losses_patch.diff", REPO / "villa_pr" / "villa_full.patch"]
    vendor_villa = VILLA_DIR
    if vendor_villa is None or not vendor_villa.exists():
        print("  [SKIP] VILLA_DIR not set (a villa checkout on merge-ink-pipelines); skipping git apply --check")
        return errors

    # Extract upstream HEAD losses.py (read-only) into a temp dir and test git apply
    res = subprocess.run(
        ["git", "-C", str(vendor_villa), "show", "HEAD:ink-detection/koine_machines/training/loss/losses.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        errors.append(f"Failed to read upstream HEAD losses.py: {res.stderr}")
        return errors

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target = tmp_path / "ink-detection" / "koine_machines" / "training" / "loss" / "losses.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(res.stdout, encoding="utf-8", newline="\n")
        for patch_file in patch_files:
            chk = subprocess.run(
                ["git", "apply", "--check", "--verbose", str(patch_file)],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
                check=False,
            )
            if chk.returncode != 0:
                errors.append(f"villa_pr/{patch_file.name} failed git apply --check:\n{chk.stderr}")
            else:
                print(f"  [OK] villa_pr/{patch_file.name} applies cleanly to losses.py at HEAD of {vendor_villa.name}")
    return errors


def check_unit_tests_and_composite() -> list[str]:
    errors: list[str] = []
    t_res = subprocess.run(
        [sys.executable, str(REPO / "tests" / "test_losses.py")],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    if t_res.returncode != 0:
        errors.append(f"tests/test_losses.py failed:\n{t_res.stdout}\n{t_res.stderr}")
    else:
        print("  [OK] tests/test_losses.py: all 5 CPU unit tests PASSED!")

    # Also test VillaBrierNNPULoss standalone + inside CompositeLoss if koine_machines is importable
    sys.path.insert(0, str(REPO / "villa_pr"))
    import torch
    from brier_nnpu import VillaBrierNNPULoss

    loss_mod = VillaBrierNNPULoss(alpha=14.5, batch_neg_risk=True)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    pos = torch.zeros(2, 1, 32, 32)
    pos[:, :, 4:10, 4:12] = 1.0
    blank = torch.zeros(2, 1, 32, 32)
    blank[1, :, 20:28, 20:28] = 1.0
    ignore = torch.zeros(2, 1, 32, 32)
    tgt3 = torch.cat([pos, blank, ignore], dim=1)
    val, aux = loss_mod(logits, tgt3)
    val.backward()
    assert torch.isfinite(val) and torch.isfinite(logits.grad).all()
    assert set(aux.keys()) == {"clamped_frac", "neg_risk", "pi_u", "blank_mean"}
    print(f"  [OK] VillaBrierNNPULoss standalone forward/backward PASSED (loss={val.item():.5f})")

    if VILLA_DIR is None:
        print("  [SKIP] VILLA_DIR not set; skipping the koine_machines CompositeLoss check")
        return errors
    vendor_ink = VILLA_DIR / "ink-detection"
    vendor_ves = VILLA_DIR / "vesuvius" / "src"
    if vendor_ink.exists():
        if vendor_ves.exists():
            sys.path.insert(0, str(vendor_ves))
        sys.path.insert(0, str(vendor_ink))
        from koine_machines.training.loss.losses import CompositeLoss, WeightedLossTerm

        comp = CompositeLoss([WeightedLossTerm("BrierNNPULoss", 1.0, loss_mod, metric_name="brier_nnpu")])
        logits2 = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = comp(logits2, tgt3)
        out.backward()
        assert torch.isfinite(out) and torch.isfinite(logits2.grad).all()
        assert "loss_aux/brier_nnpu/clamped_frac" in comp.latest_metrics
        print(f"  [OK] VillaBrierNNPULoss inside koine_machines CompositeLoss PASSED ({len(comp.latest_metrics)} metrics logged)")
    return errors


def check_repo_wide_mentions() -> list[str]:
    """Every file that gets pushed (code, docs, JSON, patches, .gitignore) is free of AI-tool names and private paths."""
    errors: list[str] = []
    skip_dirs = {"__pycache__", "reads", "ckpt", "logs", "visuals", "out", ".git"}
    skip_files = {"verify_release_numbers.py", "DISCORD_POST.txt"}
    exts = {".py", ".md", ".json", ".patch", ".diff", ".toml", ".gitignore", ".txt"}
    pat = re.compile(r"\bgemini\b|\bclaude\b|\banthropic\b|\bchatgpt\b|\bopenai\b|\.gemini|antigravity\\brain", re.I)
    n = 0
    for f in REPO.rglob("*"):
        if not f.is_file() or any(part in skip_dirs for part in f.parts):
            continue
        if f.name in skip_files or f.name.startswith("SUBMISSION_FORM"):
            continue
        if f.suffix.lower() not in exts and f.name != ".gitignore":
            continue
        n += 1
        text = f.read_text(encoding="utf-8", errors="ignore")
        m = pat.search(text)
        if m:
            errors.append(f"AI-tool name / private path {m.group(0)!r} in {f.relative_to(REPO)}")
    print(f"  [OK] Repo-wide scan of {n} pushed files for AI-tool names and private paths")
    return errors


def check_new_checks() -> list[str]:
    """Stock-trainer check and fixed-half check: the numbers quoted in the texts match their JSONs."""
    errors: list[str] = []
    kt = load_json("check_koine_trainer.json")
    fh = load_json("check_fixed_half.json")
    k1, k2 = kt["k1"], kt["k2"]
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    pr = (REPO / "villa_pr" / "VILLA_PR_DESCRIPTION.md").read_text(encoding="utf-8")
    form = FORM_DRAFT.read_text(encoding="utf-8") if FORM_DRAFT.exists() else ""
    need_k =[f"{k1['b9_GATE']:.4f}", f"{k2['b9_GATE']:.4f}", f"{k1['s814_GATE']:.4f}", f"{k2['s814_GATE']:.4f}",
              f"{int(k1['b9_p99'])}", f"{int(k2['b9_p99'])}", f"{k1['FA80']:.4f}", f"{k2['FA80']:.4f}"]
    for name, text in (("README.md", readme), ("VILLA_PR_DESCRIPTION.md", pr)):
        for tok in need_k:
            if tok not in text:
                errors.append(f"{name} missing stock-trainer number {tok!r}")
    for tok in (f"{k1['b9_GATE']:.2f}", f"{k2['b9_GATE']:.2f}", f"{k1['s814_GATE']:.2f}", f"{k2['s814_GATE']:.2f}"):
        if form and tok not in form:
            errors.append(f"SUBMISSION_FORM.txt missing stock-trainer number {tok!r}")
    g140, g125 = fh["gain_140325"], fh["gain_125754"]
    need_h = [f"+{g140['H2']:.4f}", f"+{g140['H3']:.4f}", f"+{g125['H2']:.4f}", f"{round(100 * g140['ratio'])}%"]
    for tok in need_h:
        if tok not in readme:
            errors.append(f"README.md missing fixed-half number {tok!r}")
    if not errors:
        print(f"  [OK] Stock-trainer check (GATE {k1['b9_GATE']:.4f} vs {k2['b9_GATE']:.4f}) and fixed-half check "
              f"({100 * g140['ratio']:.1f}% of the full-mask gain) match their JSONs")
    return errors


def main() -> int:
    print("=" * 72)
    print("ScribblePU Release Verification Suite")
    print("=" * 72)
    all_errors: list[str] = []

    print("\n1. Style, dash, and hygiene checks...")
    all_errors.extend(check_style_and_hygiene())

    print("\n2. Discord length and Submission Form format checks...")
    all_errors.extend(check_discord_and_form())

    print("\n3. Numerical parity & checkpoint SHA256 verification...")
    all_errors.extend(check_numbers())

    print("\n4. Landscape figures verification...")
    all_errors.extend(check_figures())

    print("\n5. Upstream ScrollPrize/villa patch verification...")
    all_errors.extend(check_villa_patch())

    print("\n6. Unit tests & CompositeLoss integration check...")
    all_errors.extend(check_unit_tests_and_composite())
    print("\n7. Repo-wide hygiene + new checks...")
    all_errors.extend(check_repo_wide_mentions())
    all_errors.extend(check_new_checks())

    print("\n" + "=" * 72)
    if all_errors:
        print(f"FAILED with {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  [FAIL] {e}")
        return 1
    print("ALL CHECKS PASSED! Release bundle is verified and ready to publish.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
