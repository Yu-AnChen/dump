S3segmenter version used: v1.5.1
- https://github.com/HMS-IDAC/S3segmenter/releases/tag/1.5.1
- https://github.com/HMS-IDAC/S3segmenter/tree/7d64fea61130446c5439159b96c144ac37873b1d/large


Local installation instruction for parameter testing
- https://github.com/HMS-IDAC/S3segmenter/blob/7d64fea61130446c5439159b96c144ac37873b1d/large/readme-watershed.md

---

Command used for segmentation
```bash
python watershed.py 
    -i "Y:\sorger\data\computation\Yu-An\YC-20221013-huan_tma_segmentation\2022_tma_1-pmap.tif" 
    -o "Y:\sorger\data\computation\Yu-An\YC-20221013-huan_tma_segmentation\2022_tma_1-f9_nucleiRing.ome.tif" 
    --maxima-footprint-size 9 
    --mean-intensity-min 80 
    --area-max 50000 
    --pixel-size 0.325
```