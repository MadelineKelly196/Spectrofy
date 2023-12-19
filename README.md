# Spectrofy
Spectrofy is a project for MLOps, Fall 2023. This project aims to deploy a webapp for song analysis using spectrographs and a CNN.

## Git Strategy

### *At the beginning of the project:*
```
git clone "https://github.com/MadelineKelly196/Spectrofy.git"
```


### *When creating a new branch:*
```
#make sure you're on main
git checkout main
git fetch origin
#remove any local changes made to main to ensure your new branch starts from main as it exists on GitHub
git reset --hard origin/main
```

Alternatively, when making a new branch:
Go to GitHub
Where you see the available branches (top left), click on the drop-down and click on "View all branches"
Then on the top right, click the blue button labeled "New Branch" and give your new branch a clear name
(Note: by default the new branch will originate from main. Keep this as it is.)

### *Checkout your feature branch*

Whenever you're working on your code, first make sure that you are working in the proper branch
```
git checkout -b my_feature_branch
```

### *Update local changes on the remote repo*

Once you've made some changes to your branch's files and you'd like to save them to the remote repo (a good idea to do frequently), then:

```
#see what files have changed from the remote version and double check which branch you're on
git status
git add <my_changed_filename1>
git add <my_changed_filename2>
git commit -m "brief message about my changes"
#push your changes from your local branch to the remote branch
git push -u origin my_feature_branch
```

### *When you're ready to merge completed features into the main branch*

From within your feature branch
```
git rebase origin/main
```

Resolve any conflicts in VSCode, if there are any. If there were issues to resolve then do the following:
```
git add <conflict_resolved_filname>
git rebase --continue
```

When the rebase is completed successfully then:
```
git push --force-with-lease
```

Now, a pull request much be made on GitHub.
1. Go to our repo on GitHub
2. Near the top left, click on Pull Requests
3. Click on the blue button labeled "New Pull Request"
4. Leave the "base" as main, but change the "compare" to the name of your feature branch
5. If it says "Able to merge" (which it should if you've resolved and successfully commited any conflicts during the rebase process), then click on the blue button labeled "Create pull request"
6. Make any notes or discussion points you'd like to have with the team at this point.
7. At least one other member of the team will need to review and approve the request. Please also review your own code and make sure things look good.
8. Resolve any suggested changes, if there are any
10. Then when approved, you can merge your branch into the main branch using "Squash and merge".
11. Once merged successfully, delete your feature branch using the Delete branch button.

When you're ready to work on the next feature, go back to the "When creating a new branch" section, to start your new feature branch.

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
    * As a rule of thumb, we install with `pip` all the packages that are not available with `conda` or are only available through the `conda-forge` channel (which makes the environment pretty slow to solve). If at some point this approach breaks the environment, either add `conda-forge` and install most of the packages with `conda`, or avoid `conda-forge` and install most of the packages with `pip` (not the recommended way for some scientific packages)

## Git LFS

1. Install with eg. `sudo apt install git-lfs` or see instructions for your system on [website](https://git-lfs.com/)
2. Set it up with `git lfs install` (to download LFS files when cloning) or with `git lfs install --skip-smudge` (to not download them)
3. That's it. PNG were already added to `.gitattributes` with `git lfs track "*.png"` (just push them as usual) and MP3 were ignored to save storage space
