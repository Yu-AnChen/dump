# Export OMERO ROIs

- Refer to https://gist.github.com/Yu-AnChen/58754f960ccd540e307ed991bc6901b0
    - Trick to quickly get multiple omero image IDs: select multiple image and
      open in PathViewer Grid - image IDs are in the URL
    - Specify which omero instance

# Mask single-cell table using exported ROIs

- ROI Coordinates -> shapes -> which cells are inside the shapes
- Generate masks for each ROI
- Logical operation of the masks
- Update single-cell table

# Visual checks and next steps

- Scatter plot (datashader is better)
- Cross-check with omero image
- Visualize intensity - log scale
- Gating
    - scimap https://github.com/labsyspharm/scimap
    - tabbi https://github.com/Yu-AnChen/tabbi