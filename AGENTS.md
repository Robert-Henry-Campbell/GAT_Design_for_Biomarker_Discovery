# Repository Guidelines

This project contains a collection of research scripts for experimenting with graph attention networks. There are no unit tests, but when modifying Python code you should ensure it compiles.

## Required Checks

Run the following command before committing if any `*.py` files were modified:

```
python -m py_compile $(git ls-files '*.py')
```

If the command fails, fix the syntax errors before opening a pull request.
