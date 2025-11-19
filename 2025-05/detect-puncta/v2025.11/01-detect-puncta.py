import pathlib
import re

import dask.array as da
import numpy as np
import palom
import pandas as pd
import skimage.morphology
import tqdm.contrib
import zarr
from spotiflow.model import Spotiflow


def process(
    img_path, out_dir, puncta_channels, dna_channel, puncta_sigma=None, model_name=None
):
    if model_name is None:
        model_name = "hybiss"
    model = Spotiflow.from_pretrained(model_name)
    if puncta_sigma is not None:
        model.config.sigma = puncta_sigma

    reader = palom.reader.OmePyramidReader(img_path)
    path = reader.path

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # reader.pyramid[0] = reader.pyramid[0][:, :10000, :10000]

    if dna_channel is not None:
        img_dna = reader.pyramid[0][dna_channel]

    imgs = reader.pyramid[0][puncta_channels]

    masks_peak = zarr.zeros(imgs.shape, chunks=imgs.chunksize, dtype="bool")

    puncta_dfs = []
    for ii, img in enumerate(imgs):
        coords, details = model.predict(
            img, subpix=False, min_distance=1, prob_thresh=None
        )
        coords = np.int32(coords)
        masks_peak.vindex[ii, coords.T[0], coords.T[1]] = True

        # -------------- save intensity properties for further filtering ------------- #
        df = pd.DataFrame()
        df[["Y", "X"]] = coords.astype("int32")
        df["prob_spotiflow"] = details.prob.astype("float32")
        # NOTE: the coordinates from spotiflow are not sorted
        df.sort_values(["Y", "X"], inplace=True)
        df["channel_intensity"] = img[
            da.from_zarr(masks_peak, name=False)[ii]
        ].compute()

        puncta_dfs.append(df)

    # ------------------------- write puncta mask to disk ------------------------ #
    stem = re.sub(r"\.ome\.tiff?$", "", path.name)
    max_digit_len = max([len(str(cc)) for cc in puncta_channels])
    channel_str = "_".join([f"{cc:0{max_digit_len}}" for cc in puncta_channels])
    out_path_mask = out_dir / f"{stem}-spots-channel_{channel_str}.ome.tif"

    palom.pyramid.write_pyramid(
        da.from_zarr(masks_peak, name=False).astype("uint8"),
        out_path_mask,
        pixel_size=reader.pixel_size,
        channel_names="",
        downscale_factor=2,
        tile_size=1024,
        save_RAM=True,
        compression="zlib",
        is_mask=True,
    )

    for cc, dd in tqdm.contrib.tzip(
        puncta_channels, puncta_dfs, desc="Writing measurements"
    ):
        out_df_name = re.sub(
            r"\.ome\.tiff?$",
            f"-measurements_{cc:0{max_digit_len}}.parquet",
            out_path_mask.name,
        )
        out_df_path = out_dir / out_df_name
        out_df_path.parent.mkdir(exist_ok=True, parents=True)

        dd.to_parquet(out_df_path, engine="pyarrow")

    # ------------------------------ write qc image ------------------------------ #
    out_qc_name = re.sub(r"\.ome\.tiff?$", "-qc.ome.tif", out_path_mask.name)
    out_qc_path = out_dir / "qc" / out_qc_name
    out_qc_path.parent.mkdir(exist_ok=True, parents=True)

    mosaics_qc = []
    channel_names_qc = []

    if dna_channel is not None:
        mosaics_qc.append(img_dna)
        channel_names_qc.append(f"{dna_channel}_DNA")

    mosaics_qc.append(imgs)
    channel_names_qc.append([f"{cc}" for cc in puncta_channels])

    for mm in da.from_zarr(masks_peak, name=False):
        mask_dilated = mm.map_overlap(
            skimage.morphology.binary_dilation,
            footprint=skimage.morphology.disk(1),
            depth=4,
            boundary=False,
        )
        mosaics_qc.append(mask_dilated)
    channel_names_qc.extend([f"{cc}_puncta" for cc in puncta_channels])

    palom.pyramid.write_pyramid(
        mosaics_qc,
        out_qc_path,
        pixel_size=reader.pixel_size,
        channel_names=channel_names_qc,
        downscale_factor=2,
        tile_size=1024,
        save_RAM=True,
        compression="zlib",
        is_mask=False,
    )


img_paths = r"""
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\registration\LSP18303.ome.tif
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\registration\LSP18315.ome.tif
""".strip().split("\n")

out_dirs = r"""
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18303\fish-spotiflow
\\research.files.med.harvard.edu\hits\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\ORION_FISH_FT_p53_Tanjina_81925\FT_iSTIC\LSP18315\fish-spotiflow
""".strip().split("\n")


puncta_channels = [22, 24]
dna_channel = 20
puncta_sigma = None


for pp, oo in zip(img_paths[:], out_dirs[:]):
    process(pp, oo, puncta_channels, dna_channel, puncta_sigma, model_name="hybiss")
