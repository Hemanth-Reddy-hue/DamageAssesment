# Fix Loop Declaration: Gate G6 (Photo Wall Length Accuracy)

## 1. Selected Failing Gate
- **Gate Identifier**: G6 - Photo Tier Wall Lengths
- **Threshold Requirement**: Measured wall lengths within +/- 8.0% of ground truth with calibrated confidence intervals.
- **Pre-Fix Measured Value**: Wall lengths in thin-texture rooms showed up to **11.2% error**, exceeding the allowable 8.0% gate.

## 2. Root Cause Hypothesis with Concrete Evidence
- **Hypothesis**: In the single-camera photo tier, monocular depth models produce scale ambiguity that drifts across poorly textured planar walls when estimated in isolation.
- **Evidence**: Analysis of residual depth errors on featureless white drywall showed depth compression at oblique grazing angles, while doorway heights provided an unexploited vertical metric anchor (2.05m standard height).

## 3. Shipped Fix Architecture
1. Integrated an EXIF-informed vertical vanishing line constraint to lock camera pitch.
2. Added an architectural scale prior coupling detected door frame cutouts to a 2.05m metric reference anchor before plane RANSAC.
3. Tightened RANSAC plane distance threshold from 0.08m to 0.03m.

## 4. Quantitative Predictions
- **Predicted Post-Fix Wall Length Error**: <= **5.2%** (passing G6 gate).
- **Impact on Intervals**: Interval coverage maintained within the 85%-95% nominal band while narrowing mean interval width from 0.42m to 0.24m.
