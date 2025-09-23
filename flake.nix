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
            url = "https://files.pythonhosted.org/packages/96/31/2953091ccc0432c6b238f01031f845f35ffe65e0f0dbb77ceda5a31978dc/pyqt-toast-notification-1.3.3.tar.gz";
            sha256 = "595dbf4b9edee77329e2514255d9a9415ff5d70708c50790db0fee207d323a29";
          };
          format = "setuptools";
          doCheck = false;
          buildInputs = [];
          checkInputs = [];
          nativeBuildInputs = [];
          propagatedBuildInputs = [
            pyPkgs.qtpy
          ];
        };
        "viztracer" = pyPkgs.buildPythonPackage {
          pname = "viztracer";
          version = "1.0.4";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/f8/11/3e42af9884046efd7c4b08f9e84d4c8b13953ba4761e67e39a95239ed02c/viztracer-1.0.4.tar.gz";
            sha256 = "1mrwx1ynz0f6s75y6adm4dp9c87s9g4s4vv77gci32fnffm6znp8";
          };
          format = "setuptools";
          doCheck = false;
          buildInputs = [];
          checkInputs = [];
          nativeBuildInputs = [];
          propagatedBuildInputs = [
            pyPkgs.objprint
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
        dependencies = (getPkgs project.dependencies) ++ (with pkgs; [
          ffmpeg-full
        ]);
      };
      editablePackage = pyPkgs.mkPythonEditablePackage {
        pname = project.name;
        inherit (project) version scripts;
        root = "$PWD/src";
        dependencies = getPkgs project.optional-dependencies.dev;
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
          build-cython() { python setup.py build_ext -j 4 --inplace; }
        '';
      };
    }
  );
}
