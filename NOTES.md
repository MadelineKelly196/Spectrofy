## conda

* Exporting environment:
  ```
  conda env export --from-history > environment.yml
  ```
  * `--from-history` doesn't include pip packages. Add them [manually](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually)

* Recreating environment:
  ```
  conda env create -f environment.yml
  ```
  * [Then](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) `pip install spotipy`, etc. (only if you forgot to add pip packages)
    * As a rule of thumb, we install with `pip` all the packages that are not available with `conda` or are only available through the `conda-forge` channel (which makes the environment pretty slow to solve). If at some point this approach breaks the environment, either add `conda-forge` and install most of the packages with `conda` or avoid `conda-forge` and install most of the packages with `pip` (not the recommended way for some scientific packages)

## Git LFS

1. Install with eg. `sudo apt install git-lfs` or see instructions on [website](https://git-lfs.com/)
2. Set it up with `git lfs install` (just once)
3. That's it. PNG were already added to `.gitattributes` with `git lfs track "*.png"` (just push them as usual) and MP3 were ignored to save storage space
4. When you clone ... TBD
