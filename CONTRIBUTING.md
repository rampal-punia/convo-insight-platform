# Contributor's Guide to the Convo Insight Platform

Welcome to the convo-insight-platform project! We're excited that you want to contribute. This guide will walk you through the process of contributing to our GitHub repository.

## Table of Contents

1. [Setting Up](#setting-up)
2. [Making Changes](#making-changes)
3. [Submitting Your Contribution](#submitting-your-contribution)
4. [After Submission](#after-submission)
5. [For Repository Owners: Reviewing and Merging Contributions](#for-repository-owners-reviewing-and-merging-contributions)

## Setting Up

1. **Create a GitHub Account**: If you don't already have one, go to [GitHub](https://github.com) and sign up for a free account.

2. **Fork the Repository**: 
   - Go to the main page of the convo-insight-platform repository.
   - In the top-right corner, click the "Fork" button.
   - This creates a copy of the repository in your GitHub account.

3. **Clone Your Fork**:
   - On your fork's page, click the "Code" button and copy the URL.
   - Open your terminal or command prompt.
   - Navigate to where you want to store the project.
   - Run: `git clone [URL you copied]`
   - This downloads the repository to your local machine.

4. **Set Up Remotes**:
   - Change into the project directory: `cd convo-insight-platform`
   - Add the original repository as a remote"
   ```
   git remote add upstream https://github.com/rampal-punia/convo-insight-platform.git
   ```
   - This allows you to keep your fork updated with the main project.

## Making Changes

1. **Create a New Branch**:
   - Create and switch to a new branch: `git checkout -b my-new-feature`
   - Use a descriptive name for your branch, like `add-python-libraries` or `fix-typo-in-readme`.

2. **Make Your Changes**:
   - Open the project in your preferred text editor or IDE.
   - Make the changes you want to contribute.
   - Save your changes.

3. **Commit Your Changes**:
   - Stage your changes: `git add .`
   - Commit the changes: `git commit -m "Add a brief, descriptive commit message"`

4. **Keep Your Fork Updated** (optional, but recommended):
   - Switch to your main branch: `git checkout main`
   - Fetch upstream changes: `git fetch upstream`
   - Merge upstream changes: `git merge upstream/main`
   - Switch back to your feature branch: `git checkout my-new-feature`
   - Rebase your branch on main: `git rebase main`

## Submitting Your Contribution

1. **Push Your Changes**:
   - Push your branch to your fork: `git push origin my-new-feature`

2. **Create a Pull Request**:
   - Go to your fork on GitHub.
   - Click on "Pull requests" and then the "New pull request" button.
   - Ensure the base repository is the original project and the base branch is `main`.
   - Ensure the head repository is your fork and the compare branch is your feature branch.
   - Click "Create pull request".
   - Add a title and description for your pull request, explaining your changes.
   - Click "Create pull request" again to submit.

## After Submission

- The repository owner will review your pull request.
- They may ask for changes or clarifications in the pull request comments.
- If changes are requested, make them in your local branch, commit, and push them. The pull request will update automatically.
- Once approved, your changes will be merged into the main project!

## For Repository Owners: Reviewing and Merging Contributions

As the repository owner, here's how you can review and merge contributions:

1. **Access Pull Requests**:
   - Go to your repository on GitHub.
   - Click on the "Pull requests" tab.

2. **Review the Pull Request**:
   - Click on the pull request you want to review.
   - Review the changes in the "Files changed" tab.
   - Leave comments or request changes if needed.

3. **Run Tests** (if applicable):
   - If you have automated tests, ensure they pass before merging.

4. **Merge the Pull Request**:
   - If you're satisfied with the changes and all checks pass:
     - Click the "Merge pull request" button.
     - Choose the merge method (usually "Merge pull request" for a standard merge).
     - Click "Confirm merge".

5. **Delete the Branch** (optional):
   - After merging, you can delete the contributor's branch if it's no longer needed.

6. **Update Your Local Repository**:
   - In your local repository:
     ```
     git checkout main
     git pull origin main
     ```

Remember to thank your contributors for their efforts!

---

We hope this guide helps you contribute to the convo-insight-platform project. If you have any questions, feel free to open an issue in the repository. Happy contributing!