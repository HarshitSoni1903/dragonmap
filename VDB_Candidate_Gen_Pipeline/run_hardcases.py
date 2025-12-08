import argparse
import os
import subprocess

HARD_LABELS = [
    "Hypoalbuminemia",
    "Micrognathia",
    "Macroglossia",
    "Dyspnea",
    "Ataxia",
    "Hepatomegaly",
    "Cyanosis",
    "Ascites",
    "Anosmia",
    "Failure to thrive",
    "Neutropenia",
    "Hypoglycemia",
    "Edema",
    "Pulmonary hypertension",
    "Abnormal cranial nerve morphology",
    "Abnormal heart septum morphology",
    "Dilated cardiomyopathy",
]


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--collection", default="enriched_mp_sapbert")
    parser.add_argument("--collection", default="enriched_mp_sapbert_ft")
    parser.add_argument("--metadata", default="data/enriched_mp.csv")
    parser.add_argument("--topk", type=int, default=25)
    args = parser.parse_args()

    print(f"📌 Using {len(HARD_LABELS)} hardcase labels")
    print(f"🔧 Collection: {args.collection}")
    print(f"🔧 Metadata: {args.metadata}")
    print(f"🔧 K: {args.topk}")
    print("🔧 Output dir: candidate_outputs/ (auto)\n")

    for label in HARD_LABELS:
        print("=" * 80)
        print(f"🔍 Running search for: {label}")
        print("=" * 80)

        cmd = [
            "python",
            "get_candidates_v5_csv.py",
            "--label",
            label,
            "--target_collection",
            args.collection,
            "--model",
            "sapbert",
            "--metadata",
            args.metadata,
            "--topk",
            str(args.topk),
        ]

        print("➡️ Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed for '{label}' — continuing. Error: {e}")

        print("\n\n")


if __name__ == "__main__":
    main()
