"""AzSLD → MediaPipe landmarks → signer-independent benchmark.

Shared code for the Kaggle notebooks in ml/notebooks/ (they embed this package
verbatim, see ml/build_notebooks.py). Phases 0–1 of the model plan:

  zenodo     read the AzSLD Zenodo archives by byte range (no 44 GB download)
  landmarks  MediaPipe hands + pose + face, same .task models as the browser
  store      landmark arrays packed into part-*.npz files + manifest.csv
  sources    dataset listings: Sentences annotations, Words_200, Fingerspelling
  split      pseudo-signer clustering and the frozen train/val/test split
  pipeline   the steps the notebooks run
"""
