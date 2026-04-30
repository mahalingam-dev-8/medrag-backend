#!/usr/bin/env python3
"""Create synthetic sample medical text files for testing.

Usage:
    python scripts/seed_sample_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_DOCS = {
    "pharmacology_basics.txt": """\
PHARMACOLOGY BASICS

1. Drug Absorption
Drug absorption refers to the process by which a drug enters the bloodstream.
Oral bioavailability depends on first-pass metabolism in the liver.
Intravenous administration provides 100% bioavailability.

2. Metformin
Metformin is the first-line pharmacological treatment for Type 2 diabetes mellitus.
It works by reducing hepatic glucose production and improving insulin sensitivity.
Common side effects include nausea, diarrhea, and abdominal discomfort.
Rarely, lactic acidosis may occur, particularly in patients with renal impairment.

3. Beta-Blockers
Beta-blockers competitively block beta-adrenergic receptors, reducing heart rate
and blood pressure by inhibiting the effects of catecholamines such as epinephrine.
They are used in hypertension, angina, and heart failure management.
""",
    "clinical_guidelines.txt": """\
CLINICAL GUIDELINES SUMMARY

Diabetes Management
- HbA1c target: <7% for most adults with Type 2 diabetes
- First-line therapy: Metformin (unless contraindicated)
- Add second agent if HbA1c not achieved after 3 months
- Consider SGLT2 inhibitors or GLP-1 agonists for cardiovascular risk reduction

Hypertension
- Target blood pressure: <130/80 mmHg for most adults
- First-line agents: ACE inhibitors, ARBs, thiazide diuretics, CCBs
- Lifestyle modifications: DASH diet, exercise, sodium restriction, weight loss

Antibiotic Stewardship
- Prescribe antibiotics only when bacterial infection is confirmed or highly suspected
- Use narrow-spectrum agents when possible
- Document indication, dose, and duration at time of prescribing
""",
}


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_DOCS.items():
        path = raw_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"Created: {path}")

    print(f"\nSample documents written to {raw_dir}/")
    print("Run: python scripts/ingest_docs.py data/raw/")


if __name__ == "__main__":
    main()
