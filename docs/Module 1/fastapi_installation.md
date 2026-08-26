# Installing FastAPI with `uv` Package Manager

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python. Combined with the `uv` Package Manager, setting up FastAPI is straightforward and efficient.

---

## Steps to Install FastAPI Using `uv`

### 1. Initialize Your Project

Start by creating a new project directory and initializing it with `uv`:

```bash
uv init sps_genai
```

This will create a project directory called `sps_genai` with a pre-configured virtual environment managed by `uv`.

### 2. Navigate to Your Project Directory

Move into the newly created project directory:

```bash
cd sps_genai
```

### 3. Install FastAPI

Use UV to install FastAPI as a dependency:

```bash
uv add fastapi --extra standard
```

### 4. Verify Installation

Ensure that FastAPI and Uvicorn are installed successfully:

```bash
uv tree
```

This command will display a list of installed dependencies, including FastAPI.

---

## Running Your FastAPI Application

1. Edit the file named `main.py` in the `sps-genai` directory:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}
```

2. Run the application using Uvicorn:

```bash
uv run fastapi dev
```

3. Open your browser and navigate to `http://127.0.0.1:8000` to see your API in action.

---

## Additional Tools and Features

### 1. Auto-Documentation

FastAPI automatically generates interactive API documentation. After running your app, you can access the following:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 2. Adding More Dependencies

To add additional dependencies (e.g., `Jupyter` or `Pydantic`):

```bash
uv add jupyter pydantic
```

### 3. Managing Your Project

`uv` helps manage your virtual environment and dependencies efficiently:

- To remove a package: `uv remove package-name`
- To sync the project's dependencies with the environment: `uv sync`
