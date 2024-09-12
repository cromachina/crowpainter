# Crowpainter
The goal for this project is to make a drawing program that is similar in feel, functionality, and performance to Paint Tool SAI, but is also multi-platform, open source, and has support for scripts and plugins.

# Developer setup
- If using Nix:
    - Run `nix develop`
- Otherwise you have to install some dependencies manually:
    - Install a C compiler (needed for Cython files)
    - Install Python 3.12
    - Install poetry into a venv, instructions here: https://python-poetry.org/docs/
- Run `poetry install` to get Python dependencies and build Cython files `.pyx` (repeat after changing any Cython files).

# Running/Debugging developer build
- From a CLI:
    - Run `poetry run crowpainter`
    - If using Nix, you can also run `nix run`
- Debugging/developing with vscode:
    - If using Nix, start vscode from the repo directory after entering the dev shell: `code .`
    - A launch config is already included for this repo.
    - I recommend turning on `User Uncaught Exceptions` for the debugger to get exceptions in async code.

# TODO
- Add all of the basic paint program stuff (way too many things to list)
- Add github workflow to make prebuilt executables.
- Add interesting advanced functions:
    - Timelapse recorder
    - Mesh transform
