import pathlib

template = """
---
title: Orion {} - overview - Lin, Chen, Campton et al., 2023
image: https://labsyspharm.github.io/orion-crc/minerva/{}/thumbnail.jpg
date: '2008-03-01'
minerva_link: https://labsyspharm.github.io/orion-crc/minerva/{}/index.html
info_link: null
show_page_link: false
tags:
    - overview-crc
---
"""

in_dir = pathlib.Path('/Users/yuanchen/projects/orion-crc/docs/minerva')
out_dir = pathlib.Path('/Users/yuanchen/projects/harvardtissueatlas/_data-cards/lin-chen-campton-2023')

overviews = sorted(in_dir.glob('P37*'))

for oo in overviews:
    name = oo.name.split('-')[-1]
    yml = template.strip().format(name, oo.name, oo.name)
    with open(out_dir / f"{oo.name.lower()}-overview.md", 'w') as f:
        f.write(yml)



import datetime
import pathlib
import yaml

from slugify import slugify

out_dir = pathlib.Path('/Users/yuanchen/projects/harvardtissueatlas/_data-cards/gray-2023')

template = """
---
title: {}
image: https://www.cycif.org/assets/img/gray-2023/{}
date: {}
minerva_link: {}
info_link: null
show_page_link: false
tags:{}
---
"""

tag_brca = """
    - Gray
    - BRCA
"""

tag_STIC = """
    - Gray
    - STIC
"""

start_date = datetime.date(2010, 2, 1)

with open('/Users/yuanchen/projects/cycif.org/_data/config-gray-2023/index.yml') as f:
    data = yaml.load(f, yaml.SafeLoader)

groups = data['group object']
total = sum([len(vv) for vv in groups.values()])

start_date += datetime.timedelta(days=total-1)

for kk, vv in groups.items():
    if 'brca' in kk.lower():
        tag = tag_brca
    else:
        tag = tag_STIC
    for ii in vv:
        out_str = template.format(
            ii['title'],
            ii['thumbnail file name'],
            start_date.isoformat(),
            ii['links'][0]['CyCIF image'],
            tag
        )
        start_date += datetime.timedelta(days=-1)
        
        with open(out_dir / f"{slugify(ii['title'])}.md", 'w') as f:
            f.write(out_str.strip())
