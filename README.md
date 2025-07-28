# Crowpainter
The goal for this project is to make a drawing program that is similar in feel, functionality, and performance to Paint Tool SAI, but is also multi-platform, open source, and has support for scripts and plugins.

### Installation from source
- Install python: https://www.python.org/downloads/
- Install project: `pip install -e .`
- Run with: `crowpainter`
- If exporting doesn't work, you may also need to install ffmpeg: https://www.gyan.dev/ffmpeg/builds/#release-builds
  - You can add the bin directory to your path, or copy ffmpeg.exe to the script folder.

### Building/installing with Nix
- This project is a Nix flake, so you can run flake commands to interact with the package `nix run`, `nix build`, etc.
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
