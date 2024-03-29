import tifffile
import argparse
import sys


def fix_stitcher_ome_xml(
    img_path,
    replace_from,
    replace_to
):
    ori = tifffile.tiffcomment(img_path)
    n_to_replace = ori.count(replace_from)
    if n_to_replace == 0:
        print(f"Substring to be replaced not found in the file ({img_path})")
        return 0
    fixed = ori.replace(replace_from, replace_to)
    tifffile.tiffcomment(img_path, fixed.encode())
    print(f"{n_to_replace} instance(s) of {replace_from} replaced with {replace_to}")
    return 0


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(
        description=(
            'Fix invalid ome-xml by string replacement'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'filepath',
        help='path of input ome-tiff',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-f',
        metavar='replace-string-from',
        help='String to be replaced',
        default='</Channel><Plane',
        required=False,
    )
    parser.add_argument(
        '-t',
        metavar='replace-string-to',
        help='New string',
        default='</Channel><MetadataOnly></MetadataOnly><Plane',
        required=False
    )
    args = parser.parse_args(argv[1:])
    return fix_stitcher_ome_xml(args.filepath, args.f, args.t)


if __name__ == '__main__':
    sys.exit(main())
    '''
    Example:
    python fix-ome-xml.py input-image.ome.tiff
    '''
