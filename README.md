# Exploring oxidation states in MOFs

## Goals
* mine annotated oxidation states from the CSD

* explore them

* build a ML model to predict them to e.g. initialize DFT


## Mining
* a lot of structures contain oxidation states in the names, this annotation is done by chemists and probably correct

* we use a regex to find the oxdiation states

* need to export `CSD_HOME='/home/kevin/CCDC/CSD_2019` on thinkpad

## ML
* the descriptor should be local, focus on the relevant metal site

* should consider distances and also geometry (e.g. square planar d8 is probably different)

* should consider the chemical nature of the neighbors (e.g. electronegativity)

* a fingerprint function is probably too much in terms of dimensionality and effort, likely bond order descriptors are
  enough with local electronegativity differences. This is inspired by the bond valence method.
