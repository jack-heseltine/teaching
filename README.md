# Neural Networks for Students

🧠

This repository is a small teaching project about how neural networks learn. It is written for classroom use and is best explored step by step.

The main file is [IAT/iat_demo.ipynb](IAT/iat_demo.ipynb). If you are new to the topic, start there.

## What You Will See

The notebook builds up the main ideas behind neural networks in a simple order:

1. A straight line as a model
2. Error and residuals
3. Derivatives as a guide toward a better solution
4. Gradient descent
5. A computation graph with `micrograd`
6. Linear regression with one and two parameters
7. Classification with one neuron
8. Classification with a small neural network
9. A final step toward MNIST digit recognition

The goal is not just to use machine learning, but to understand what the model is doing.

## Setup in VS Code

💻

If this is your first time running the project, follow these steps in order.

### 1. Open the folder in VS Code

Open VS Code and open the whole project folder:

- File -> Open Folder
- choose the `teaching` folder

It is best to open the whole folder, not just a single file, so VS Code can find the notebook, script, data folder, and Python environment correctly.

### 2. Install Python and VS Code extensions

Make sure Python is installed on your computer.

In VS Code, install these extensions:

- Python
- Jupyter

These let you run `.py` files and notebooks directly inside VS Code.

### 3. Create a virtual environment

📦

A virtual environment keeps this project's packages separate from other Python projects.

Open a terminal in VS Code and run:

```bash
python3 -m venv .venv
```

If that does not work, try:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal usually shows `(.venv)` at the beginning of the line.

### 5. Install the packages

⬇️

With the virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

This may take a few minutes because it includes notebook tools, plotting libraries, and machine learning packages.

### 6. Tell VS Code to use the virtual environment

In VS Code:

- open the Command Palette with `Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows
- search for `Python: Select Interpreter`
- choose the interpreter inside `.venv`

If you are using the notebook, also check the kernel in the top right of the notebook window and select the same `.venv` environment.

### 7. First run of the notebook

Now open [IAT/iat_demo.ipynb](IAT/iat_demo.ipynb).

For the smoothest first run:

1. Run the cells from top to bottom.
2. Wait for each cell to finish before starting the next one.
3. If a later cell fails, rerun the earlier helper cells first.

### 8. If something goes wrong

🛠️

Here are the most common fixes:

- `ModuleNotFoundError`: the packages are not installed in the selected environment. Activate `.venv` and run `pip install -r requirements.txt` again.
- Notebook runs with the wrong Python: change the notebook kernel to `.venv`.
- A variable or function is missing: rerun the earlier cells because notebooks depend on execution order.
- Graph drawing problems: make sure `graphviz` was installed from `requirements.txt`.

## Best Place To Start

🚀

Open [IAT/iat_demo.ipynb](IAT/iat_demo.ipynb) in VS Code or Jupyter and run the cells from top to bottom.

Many cells include animations and visualizations. These are there to help you build intuition:

- how a model makes predictions
- how error changes
- how parameters move during learning
- how a neural network separates classes

If a cell fails, it is often enough to run the earlier cells again so that helper functions and variables are loaded.

## Other File

🔢

[IAT/mnist-pytorch.py](IAT/mnist-pytorch.py) is a separate PyTorch script for classifying handwritten digits from the MNIST dataset. Think of it as the more advanced follow-up after the notebook.

## What You Need

This project uses Python with common data science libraries such as Jupyter, NumPy, Matplotlib, scikit-learn, PyTorch, Graphviz, and `micrograd`.

There is also a [requirements.txt](requirements.txt) file in the repo root.

## Suggested Student Workflow

1. Open the notebook.
2. Run one section at a time.
3. Pause after each plot or animation and explain in your own words what changed.
4. Change a few values and see what happens.
5. Move to the MNIST script only after the notebook ideas make sense.

## Big Idea

✨

Neural networks are not magic. They are built from simple mathematical steps repeated many times. This notebook is designed to make those steps visible.