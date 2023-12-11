* Exporting environment:
  ```
  conda env export --from-history > environment.yml
  ```
  * `--from-history` doesn't include pip packages. Add them [manually](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually)
* Recreating environment:
  ```
  conda env create -f environment.yml
  ```
  * [Then](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) `pip install spotipy` (only if you forgot to add pip packages)
