---

## Setting Up a GitHub Repository

To manage your project with GitHub, follow these steps:

### 1. Install Git (if not already installed)

Download and install Git from [here](https://git-scm.com/downloads). After installation, verify by running:
```bash
git --version
```
This should return the installed Git version.

### 2. Initialize a Git Repository

This is usually done with the
```bash
git init
```
command, but since we created this project with `uv init` command, uv has automatically done this for you.


### 3. Add a `.gitignore` File

Git uses `.gitignore` file to specify which files should not be committed to a repository for version control.  These are usually large intermediate files, as well as files that contain potentially sensitive information, such as API keys.

Since you created the uv project using `uv init` command, a `.gitignore` file has automatically been created for you.

At the end of this file add `.env` line to also exclude the `.env` file that we will use to store various configurations and API keys.  This file could contain sensitive information and should not be committed to GitHub.

### 4. Commit Your Initial Files

Stage and commit your project files:
```bash
git add .
git commit -m "Initial commit - FastAPI setup with UV"
```

### 5. Create a New Repository on GitHub

Go to [GitHub](https://github.com) and create a new repository. Copy the repository URL.

### 6. Add the Remote Repository and Push Code

Run the following command, replacing `<your-repo-url>` with your GitHub repository URL:
```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

### 7. Verify Your Repository

Go to your GitHub repository page and confirm that your files have been uploaded.

---
