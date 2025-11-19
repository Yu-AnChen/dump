import pathlib
import pandas as pd


slide_dirs = r"""
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315
""".strip().split("\n")

for slide_dir in slide_dirs:
    slide_dir = pathlib.Path(slide_dir)

    quant_path = next(slide_dir.glob("quantification/*-cell.csv"))
    fish_path = next(slide_dir.glob("fish-spotiflow/counts/cellpose-nucleus.parquet"))
    out_path = fish_path.parent / f"{slide_dir.name}-quantification-fish.csv"

    print("writing to:", out_path)

    quant_columns = [
        "CellID",
        "Hoechst-ORION",
        "AF1",
        "MX1",
        "Ki67",
        "CD3e",
        "FOXJ1",
        "p21",
        "PAX8",
        "CD68",
        "CD11c",
        "Ecadherin",
        "MDM4",
        "CD4",
        "gH2ax",
        "CCNB1",
        "CD31",
        "p53",
        "53BP1",
        "CD8a",
        "Hoechst-FISH",
        "CEP1fish",
        "MDM4fish",
        "X_centroid",
        "Y_centroid",
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "Solidity",
        "Extent",
        "Orientation",
    ]

    pd.read_csv(
        quant_path, index_col="CellID", engine="pyarrow", usecols=quant_columns
    ).join(pd.read_parquet(fish_path, engine="pyarrow")).fillna(0).round(2).to_csv(
        out_path
    )
