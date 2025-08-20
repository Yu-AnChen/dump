import pathlib
import pandas as pd


for dd in ["LSP17773", "LSP33836_P110"]:
    slide_dir = pathlib.Path(rf"T:\ORION_FISH_FT_fullpanel\{dd}")

    quant_path = next(slide_dir.glob("quantification/*_cellRing.csv"))
    fish_path = next(slide_dir.glob("fish/counts/nucleiRing.parquet"))
    out_path = fish_path.parent / f"{slide_dir.name}-quantification-fish.csv"

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
        "E-cadherin",
        "MDM4",
        "CD4",
        "gH2ax",
        "CCNB1",
        "CD31",
        "p53",
        "53BP1",
        "CD8a",
        "Vimentin",
        "Hoechst-FISH",
        "Green-CEP1",
        "Orange-MDM4",
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
    ).join(
        pd.read_parquet(fish_path, engine="pyarrow").rename(
            columns=lambda x: f"fish_{x}" if "count" not in x else x
        )
    ).fillna(0).round(2).to_csv(out_path)
