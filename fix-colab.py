import nbformat

notebook_path = "notebooks/week3_finetuning.ipynb"

with open(notebook_path, "r") as f:
    nb = nbformat.read(f, as_version=4)

# Remove the broken widget metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(notebook_path, "w") as f:
    nbformat.write(nb, f)

print("Fixed.")