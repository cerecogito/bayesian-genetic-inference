# Data (all synthetic)

Every name and trait here is invented. None of it is real family or medical data.

| File | People | Observed traits | Purpose |
|---|---|---|---|
| `synthetic_pedigree_8.csv` | 8 (4 founders, 2 children, 2 grandchildren) | 3 (Alan = 1, Fiona = 0, Grace = 1) | main enumeration experiment: 3^8 × 2^5 = 209,952 configurations |
| `small_family_3.csv` | 3 (two founders and their child) | 2 (James = 1, Lily = 0) | readable example; same structure as the CS50 Heredity example family, so its output can be compared with the published numbers |

Format (same as CS50 AI Heredity): `name,mother,father,trait`. Parents are blank for founders.
`trait` is `1` (has the trait), `0` (does not) or blank (unknown).

Family tree of the 8-person pedigree:

```
Alan ─┬─ Beth        Carl ─┬─ Dana
      │                    │
     Evan ────────┬────── Fiona
                  │
           ┌──────┴──────┐
         Grace         Henry
```
