{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShell.${system} = pkgs.mkShell {
        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            sdl3
            libGL
          ]);
          NIX_ENFORCE_NO_NATIVE = "0";
        };
        inputsFrom = [
        ];
        buildInputs = with pkgs; [
          sdl3
          libGL
          pkg-config
          nim
          #nimlangserver
          nimble
        ];
      };
    };
}
