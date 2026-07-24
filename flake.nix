{
  description = "A basic Nix Flake for Rust-based agent development";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, utils, rust-overlay, ... }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # Dependencies for development that are not system packages, but still required for development (eg; z3 and JDK)
        DevDependencies = with pkgs; [
          tmux
          python3
          ripgrep
          z3
          gh
          drawio
          opencode
        ];

        # System libraries go here (e.g. openssl, pkg-config)
        MedievalDependencies = with pkgs; [
          clang-tools
          libclang
          llvmPackages.libclang
          pkg-config
          openssl
        ];

        # rust-specific dependencies
        RustDependencies = with pkgs; [
          cargo
          rustc
          rust-analyzer
          rustfmt
          clippy
          rust-analyzer
        ];

        ollamaModels = with pkgs; [
          "gemma4:e4b"
        ];

        clangMkShell = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; };

      in
      {
        devShells.default = clangMkShell {
          buildInputs = with pkgs; [
            ollama
          ] ++ MedievalDependencies ++ DevDependencies ++ RustDependencies;

          # Fixes rust-analyzer looking for standard library source code
          RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
          LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib";
          shellHook = ''
            export PATH="${pkgs.clangStdenv.cc}/bin:$PATH";
          '';
        };
      });
}

