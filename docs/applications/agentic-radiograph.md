# Match a reference radiograph pose

Give the agent a CT and a reference radiograph, including one from a different patient. It compares candidate orientations and adjusts the CT pose to reduce image loss. The result is the best-scoring object-to-world transform visited during the search.

The CT can be a NIfTI file or a DICOM series directory. The radiograph can be PNG, JPEG, TIFF, a two-dimensional NumPy array or a single-frame DICOM image. The example compares image structure using normalised cross-correlation and tests both brightness polarities. Anatomical differences remain in the residual; the objective is a useful pose match with a measurable loss reduction.

## Run the example

From a source checkout with CUDA available:

```bash
python -m pip install '.[gpu,imaging,examples]'

python examples/agentic_radiograph.py inspect scan.nii.gz radiograph.png
python examples/agentic_radiograph.py fit scan.nii.gz radiograph.png \
  --output outputs/radiograph-fit --starts 3 --iterations 80
```

Replace `scan.nii.gz` with a DICOM directory to use a series. Add `--series-uid UID` if the directory contains several series. CT values default to HU.

The image is resized to a maximum side of 192 pixels for fitting, preserving its aspect ratio. Use `--max-size` to change that limit, or `--max-size 0` to retain the original resolution. Each run requires a new output directory.

## Approximate camera geometry

The initial camera is centred on the CT. Source and detector distances and detector coverage are derived from the volume's physical extent. Pixels are square; detector rows follow the radiograph's top-to-bottom order.

When detector pitch is unknown, the tool compares six fields of view at eight axial orientations, then refines the best three combinations. This lets it account for image scale before adjusting the pose. At the default settings, the search uses at most 291 projections and 240 backward evaluations.

Supply approximate dimensions when available:

```bash
python examples/agentic_radiograph.py fit scan.nii.gz radiograph.png \
  --sod-mm 1000 --sdd-mm 1500 --pixel-spacing-mm 0.4 \
  --output outputs/radiograph-fit
```

`--pixel-spacing-mm` refers to the original radiograph and fixes the field of view. The tool adjusts the detector pitch when resizing the image. The result records the selected camera setting alongside the estimated pose. The transform is expressed in RAS millimetres under that camera model.

## Give the task to an agent

> Inspect `scan.nii.gz` and `radiograph.png`. The radiograph may come from another patient. Adjust the CT to match its projection pose using at most three starting candidates and 80 iterations per candidate. Retain the lowest-loss pose encountered. Write to a new output directory. Read the initial and final loss, candidate scores, transform and camera settings, then inspect the fitted projection and overlay. Report the loss reduction and selected pose, with the paths to the comparison image and scored trajectory.

The agent calls `inspect`, then `fit`. Both return JSON. The fit retains the highest-scoring pose visited during the search and records the alternatives it evaluated.

## Inspect the result

The output includes the resolved rigid transform, assumed camera geometry, input hashes, initial and best loss, candidate scores and a trajectory of evaluated poses. It also saves the computed optical depth and images for comparison with the supplied radiograph. If the search finds no improvement, it retains the initial best candidate and reports that outcome.

| File | Contents |
|---|---|
| `result.json` | Selected pose and camera, loss reduction and evaluation counts |
| `candidates.json` | Scores for the starting poses and camera hypotheses |
| `trajectory.json` | Every evaluated pose and its image score |
| `optical_depth.npy` | Projection at the selected pose |
| `comparison.png` | Reference radiograph beside the fitted projection |
| `overlay.png` | Reference in magenta and fitted projection in green |

The matching score is absolute normalised cross-correlation. Higher scores indicate closer image structure; the signed correlation identifies the selected brightness polarity. A single projection gives less information about translation along the viewing direction than about motion across the detector.

For a case that generates two radiographs from a supplied CT, use the [two-view example](agentic-imaging.md).
