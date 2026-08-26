# Virtual Environment and Package Manager

For all of the python code we will use the `uv` package and project manager.

## **Installation (using a terminal window)**

On macOS and Linux:


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

exec $SHELL
```

On Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or follow instructions at https://docs.astral.sh/uv/.

**Verify Installation** 

After installation, check if `uv` is installed correctly:
```bash
uv --version
```

---

## **Python Installation**

If you are cloning an existing uv project from GitHub, uv will check that the required version of python (specified in pyproject.toml file) is installed or will install it automatically.

You can also install a specific version using
```bash
uv python install 3.12
```

## **Why Use `uv` as a Package Manager?**

`uv` is a modern Python package and environment manager designed to simplify dependency management, project isolation, and build processes for Python developers.

### **1. Simplified Dependency Management**

- **Unified Workflow**: Unlike `pip` and `requirements.txt`, `uv` uses a declarative `pyproject.toml` format that combines dependency tracking, environment setup, and build configuration in a single file.
- **Automatic Locking**: Dependencies are automatically locked to ensure deterministic builds, making your project reproducible across different environments.
- **Easier Add/Remove Operations**: Adding or removing dependencies is as simple as:
  ```bash
  uv add fastapi
  uv remove fastapi
  ```
### **2. Environment Isolation**

- **Integrated Virtual Environments**: UV automatically creates and manages a virtual environment for your project. There’s no need to use venv or virtualenv separately.

- **No Global Pollution**: All project dependencies are isolated within the `.uv` environment, preventing conflicts between projects.

### **3. Developer Productivity**:

- **Streamlined Initialization**: Quickly start a new project using:
  ```bash
  uv new project-name
  ```
This sets up the directory structure, environment, and initial configuration in one step.

### **4. Enhanced Docker Integration**:

- **Simplifies Docker Builds**: When using Docker, uv streamlines dependency installation and environment setup within your container. A typical Dockerfile using uv handles both installation and setup in fewer lines compared to traditional approaches.
- **Fewer Dependency Conflicts**: Deterministic builds ensure that your Dockerized app behaves consistently, even when running on different machines.


### **5. Future-Proofing**:

- **PEP 517 and PEP 518 Compliance**: uv fully embraces modern Python packaging standards, making it compatible with the latest advancements in the Python ecosystem.
- **Modular and Extensible**: The tool is designed to evolve with modern developer needs, ensuring long-term usability.


