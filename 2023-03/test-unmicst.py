import pathlib
import subprocess

out_dir = pathlib.Path().absolute() / 'unmicst-test-data'
out_dir.mkdir(exist_ok=True)

if not (out_dir / 'in.ome.tif').exists():
    print('Downloading test image to', out_dir, '\n')
    subprocess.run([
        'curl',
        '-f',
        '-o',
        out_dir / 'in.ome.tif',
        'https://mcmicro.s3.amazonaws.com/ci/exemplar-001-cycle6.ome.tif'
    ])


for i in range(2):
    subprocess.run(
        'singularity run --nv -B'.split(' ') +
        [f"{out_dir}:/data"] +
        f'/n/groups/lsp/mcmicro/singularity/labsyspharm-unmicst-2.7.5.img python /app/unmicstWrapper.py --stackOutput --outputPath /data/run{i:02} /data/in.ome.tif'.split(' ')
    )

cmd = 'singularity run /n/groups/lsp/mcmicro/singularity/labsyspharm-unmicst-2.7.5.img python -c'

script = '''
import numpy as np
import tifffile
p1 = tifffile.imread('/data/run00/in_Probabilities_1.tif')
p2 = tifffile.imread('/data/run01/in_Probabilities_1.tif')
abs_diff = np.abs(p1.astype(int)-p2.astype(int))
print('N different pxs:', np.sum(abs_diff != 0))
print('N absolute differences >= 2:', np.sum(abs_diff >= 2))
print('N absolute differences >= 3:', np.sum(abs_diff >= 3))
'''

subprocess.run(
    'singularity run --nv -B'.split(' ') +
    [f"{out_dir}:/data"] +
    '/n/groups/lsp/mcmicro/singularity/labsyspharm-unmicst-2.7.5.img python -c'.split(' ') +
    [';'.join(script.strip().split('\n'))]
)
