import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os

notebook_file = sys.argv[1]
print(f"Executing {notebook_file}...")

with open(notebook_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
cwd = os.path.dirname(os.path.abspath(notebook_file))

ep.preprocess(nb, {'metadata': {'path': cwd}})

with open(notebook_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Successfully executed and saved {notebook_file}!")
