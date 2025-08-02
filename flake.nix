{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      pyPkgs = pkgs.python313Packages // {
        "pyqt-toast-notification" = pyPkgs.buildPythonPackage {
          pname = "pyqt-toast-notification";
          version = "1.3.3";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/84/3a/7614af8234f36a38476ae62599ba666b2bda66a71605323a4dbadcf94776/pyqt_toast_notification-1.3.3-py3-none-any.whl";
            sha256 = "05adrnqzb5ywy7b6hd0fdrzs2y5mn8m4afmdngq2i9vaacgh3fk5";
          };
          format = "wheel";
          doCheck = false;
          buildInputs = [];
          checkInputs = [];
          nativeBuildInputs = [];
          propagatedBuildInputs = [
            pyPkgs.qtpy
          ];
        };
      };
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      project = pyproject.project;
      fixString = x: lib.strings.toLower (builtins.replaceStrings ["_"] ["-"] x);
      getPkgs = x: lib.attrsets.attrVals (builtins.map fixString x) pyPkgs;
      package = pyPkgs.buildPythonPackage {
        pname = project.name;
        version = project.version;
        format = "pyproject";
        src = ./.;
        build-system = getPkgs pyproject.build-system.requires;
        dependencies = getPkgs project.dependencies ++ [ pkgs.ffmpeg-full ];
      };
      editablePackage = pyPkgs.mkPythonEditablePackage {
        pname = project.name;
        version = project.version;
        scripts = project.scripts;
        root = "$PWD/src";
      };
    in
    {
      packages.default = pyPkgs.toPythonApplication package;
      devShells.default = pkgs.mkShell {
        inputsFrom = [
          package
        ];
        buildInputs = [
          editablePackage
          pyPkgs.build
        ];
        shellHook = ''
          build-cython() { python setup.py build_ext --inplace; }
        '';
      };
    }
  );
}
