# Real-World Datasets

This directory should contain raw data files for real-world experiments. These files are not tracked in git due to their size.

## Download Instructions

### ETT Datasets (Electricity Transformer Temperature)

```bash
# From project root
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv -P data/real_world/raw/
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv -P data/real_world/raw/
```

### Sunspot Dataset (SILSO)

1. Visit: https://www.sidc.be/SILSO/INFO/snmtotcsv.php
2. Download the monthly mean sunspot number file
3. Save as: `data/real_world/raw/sunspot.csv`

## Expected Files

After downloading, this directory should contain:

```
data/real_world/raw/
├── README.md       # This file
├── ETTh1.csv      # ~2.3 MB
├── ETTh2.csv      # ~2.3 MB
└── sunspot.csv    # ~300 KB
```

## Citations

See [data/real_world/DATASETS.md](../DATASETS.md) for complete dataset descriptions and citations.
