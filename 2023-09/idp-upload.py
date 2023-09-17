template = """
tmux new-session -d -s {img_id} \; send -t {img_id} "mamba activate omeropy && omero import --exclude=clientpath --skip=upgrade --skip=checksum --skip=minmax -d 1972 {img_path} > {img_id}.log" ENTER
"""

df = pd.read_csv('/Users/yuanchen/Downloads/orion-project-files-20230515.csv')

img_ids = df['Name'][:41]
img_paths = df['Orion filepath'].apply(
    lambda x: x.replace(r'\\files.med.harvard.edu\ImStor\sorger\data\RareCyte\RareCyte-S3', '/n/files/ImStor/sorger/data/RareCyte/RareCyte-S3').replace('\\', '/')
)[:41]


with open('/Users/yuanchen/Downloads/20230916-orion-crc-idp-upload.txt', 'w') as f:

    for ii, pp in zip(img_ids, img_paths):
        ii = ii.replace(' ', '-')
        template.format(img_id=ii, img_path=pp)
        f.write(template.format(img_id=ii, img_path=pp).strip())
        f.write('\n')
