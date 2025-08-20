import argparse

import ome_types
import tifffile


def _add_channel_names(tiff_path, channel_names):
    ome = ome_types.from_tiff(tiff_path)
    n_channels = len(ome.images[0].pixels.channels)
    n_names = len(channel_names)
    assert n_channels == n_names, (
        f"Number of channels ({n_channels}) in '{tiff_path}' does not match number of channel names ({n_names})."
    )

    for channel, name in zip(ome.images[0].pixels.channels, channel_names):
        channel.name = name

    tifffile.tiffcomment(tiff_path, ome.to_xml().encode())
    return


def add_channel_names(tiff_path, names_path):
    names = open(names_path).read().strip().split("\n")
    _add_channel_names(tiff_path, names)


def main():
    parser = argparse.ArgumentParser(
        description="Add channel names to an OME-TIFF file."
    )
    parser.add_argument("tiff_path", help="Path to the OME-TIFF file.")
    parser.add_argument(
        "names_path", help="Path to a text file with one channel name per line."
    )
    args = parser.parse_args()

    add_channel_names(args.tiff_path, args.names_path)
    print(f"✅ Channel names added to: {args.tiff_path}")


if __name__ == "__main__":
    main()
