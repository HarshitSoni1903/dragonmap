# Dragon Map + CurateGPT Ontology Enrichment Pipeline

This guide reproduces the full end-to-end workflow used in DRAGON_MAP_DATA, from fetching OWL ontologies to exporting enriched, patched, and converted .owl files.

## Repository Layout
```
DRAGON_MAP_DATA/
├─ dragon_map/               # Ontology ingestion & base CSV export
│  ├─ pyproject.toml
│  └─ dragon_map/
├─ curategpt/                # Enrichment, patching & OWL export
│  ├─ pyproject.toml
│  ├─ scripts/
│  │   └─ fill_missing.py
│  ├─ out/                   # Shared output files
│  └─ stagedb/
```

## Setup Environment

### Install Dependencies
```bash
# Dragon Map
cd DRAGON_MAP_DATA/dragon_map
poetry install

# CurateGPT
cd ../curategpt
poetry install
```

### Set Environment Variables
```bash
# from DRAGON_MAP_DATA/curategpt
export OUTDIR="$PWD/out"
export STAGE="$PWD/stagedb"
mkdir -p "$OUTDIR" "$STAGE"
```

Optionally add to `~/.zshrc` so it's available next session.

## Dragon Map — Ontology Acquisition and Export

### Download Ontologies (HP & MP)
```bash
cd ../dragon_map
export DM_CACHE="$PWD/.cache"
export DM_OUTDIR="../curategpt/out"
mkdir -p "$DM_CACHE" "$DM_OUTDIR"

poetry run python -m dragon_map.download \
  --onto hp mp \
  --include-imports \
  --cache "$DM_CACHE"
```

### Export to CSV
```bash
poetry run python -m dragon_map.export_csv \
  --source "$DM_CACHE/hp.owl" \
  --out "$DM_OUTDIR/hp_enriched.csv" \
  --fields curie,label,definition,synonyms \
  --keep-imports

poetry run python -m dragon_map.export_csv \
  --source "$DM_CACHE/mp.owl" \
  --out "$DM_OUTDIR/mp_enriched.csv" \
  --fields curie,label,definition,synonyms \
  --keep-imports
```

### Verify
```bash
head ../curategpt/out/hp_enriched.csv
head ../curategpt/out/mp_enriched.csv
```


## CurateGPT — Missing Detection & Filling

### Detect Missing Definitions/Synonyms
```bash
cd ../curategpt
poetry run python - <<'PY'
import os, pandas as pd
outdir=os.environ.get("OUTDIR",os.path.join(os.getcwd(),"out"))
def missing_mask(df):
    def_missing=df['definition'].fillna("").astype(str).str.strip().eq("")
    syn_missing=df['synonyms'].fillna("").astype(str).str.strip().eq("")
    return def_missing|syn_missing
for ont in ["hp","mp"]:
    df=pd.read_csv(os.path.join(outdir,f"{ont}_enriched.csv"))
    miss=df[missing_mask(df)].copy()
    miss.to_csv(os.path.join(outdir,f"{ont}_missing_all.csv"),index=False)
    print(f"{ont.upper()} -> enriched={len(df)} missing={len(miss)}")
PY
```

### Fill Missing Values (Heuristic)

Your `scripts/fill_missing.py` already does this; minimal version:
```bash
poetry run python scripts/fill_missing.py \
  --input "$OUTDIR/hp_missing_all.csv" \
  --output "$OUTDIR/hp_missing_filled_all.csv"

poetry run python scripts/fill_missing.py \
  --input "$OUTDIR/mp_missing_all.csv" \
  --output "$OUTDIR/mp_missing_filled_all.csv"
```

**Expected results:**
```
HP  definitions: 3513 → 0   synonyms: 11341 → 11053
MP  definitions: 1333 → 0   synonyms:  9976 → 9784
```


## Merge Fills Back into Enriched CSVs
```bash
poetry run python - <<'PY'
import os,pandas as pd
outdir=os.environ.get("OUTDIR",os.path.join(os.getcwd(),"out"))
def blank(s): return s.fillna("").astype(str).str.strip().eq("")
def patch_one(ont):
    enr=pd.read_csv(f"{outdir}/{ont}_enriched.csv")
    fil=pd.read_csv(f"{outdir}/{ont}_missing_filled_all.csv")
    m=enr.merge(fil.add_prefix("filled_"),left_on="curie",right_on="filled_curie",how="left")
    def0=blank(m["definition"]).sum(); syn0=blank(m["synonyms"]).sum()
    for c in["definition","synonyms"]:
        fc=f"filled_{c}"
        if c in m and fc in m:
            take=blank(m[c]) & (~blank(m[fc]))
            m.loc[take,c]=m.loc[take,fc]
    def1=blank(m["definition"]).sum(); syn1=blank(m["synonyms"]).sum()
    m.drop(columns=[c for c in m.columns if c.startswith("filled_")],inplace=True,errors="ignore")
    m.to_csv(f"{outdir}/{ont}_enriched_patched.csv",index=False)
    print(f"{ont.upper()} defs {def0}->{def1} syns {syn0}->{syn1}")
for o in["hp","mp"]: patch_one(o)
PY
```

**Results:**
```
HP defs 3513→0  syns 11341→11053
MP defs 1333→0  syns 9976→9784
```

## Verify Completeness
```bash
poetry run python - <<'PY'
import os,pandas as pd
outdir=os.environ.get("OUTDIR",os.path.join(os.getcwd(),"out"))
for o in["hp","mp"]:
    df=pd.read_csv(f"{outdir}/{o}_enriched_patched.csv")
    d=df["definition"].fillna("").astype(str).str.strip().eq("").sum()
    s=df["synonyms"].fillna("").astype(str).str.strip().eq("").sum()
    print(f"{o.upper()} rows={len(df)} defs_missing={d} syns_missing={s}")
PY
```

**Expected:**
```
HP rows=31402  defs_missing=0  syns_missing=11053
MP rows=34291  defs_missing=0  syns_missing=9784
```

## Export Final OWL Files
```bash
poetry run python - <<'PY'
import pandas as pd, os
out=os.environ.get("OUTDIR",os.path.join(os.getcwd(),"out"))
def csv_to_owl(ont):
    df=pd.read_csv(f"{out}/{ont}_enriched_patched.csv")
    p=f"{out}/{ont}_enriched_patched.owl"
    with open(p,"w",encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f'<rdf:RDF xmlns="http://purl.obolibrary.org/obo/{ont}.owl#"\n')
        f.write(' xml:base="http://purl.obolibrary.org/obo/"\n')
        f.write(' xmlns:obo="http://purl.obolibrary.org/obo/"\n')
        f.write(' xmlns:owl="http://www.w3.org/2002/07/owl#"\n')
        f.write(' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n')
        f.write(' xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n')
        f.write(' xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"\n')
        f.write(' xmlns:IAO="http://purl.obolibrary.org/obo/IAO_">\n\n')
        for _,r in df.iterrows():
            cur=r["curie"].replace(":","_")
            lab=str(r.get("label","")).replace("&","&amp;").strip()
            de=str(r.get("definition","")).replace("&","&amp;").strip()
            syns=[s.strip() for s in str(r.get("synonyms","")).split("|") if s.strip()]
            f.write(f'  <owl:Class rdf:about="http://purl.obolibrary.org/obo/{cur}">\n')
            if lab: f.write(f'    <rdfs:label>{lab}</rdfs:label>\n')
            if de:  f.write(f'    <IAO:0000115>{de}</IAO:0000115>\n')
            for s in syns:
                f.write(f'    <oboInOwl:hasExactSynonym>{s}</oboInOwl:hasExactSynonym>\n')
            f.write('  </owl:Class>\n\n')
        f.write('</rdf:RDF>\n')
    print("Wrote",p)
for o in["hp","mp"]: csv_to_owl(o)
PY
```

**Outputs:**

- `out/hp_enriched_patched.owl`
- `out/mp_enriched_patched.owl`
