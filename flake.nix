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

        faissForRust = pkgs.stdenv.mkDerivation rec {
          pname = "faiss-c-api";
          version = "1.14.3";
          src = pkgs.fetchFromGitHub {
            owner = "facebookresearch";
            repo = "faiss";
            rev = "v${version}";
            hash = "sha256-lIyb+T3tvCqfIqUJ6KtubnLWYTlOt5Cz51mZmDW+AYo=";
          };
        
          nativeBuildInputs = [ pkgs.cmake pkgs.gfortran ];
          buildInputs = [ pkgs.blas pkgs.lapack ];
          cmakeFlags = [
            "-DFAISS_ENABLE_C_API=ON"
            "-DBUILD_SHARED_LIBS=ON"
            "-DFAISS_ENABLE_PYTHON=OFF"
            "-DFAISS_ENABLE_GPU=OFF"
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
          ];
        };

        # System libraries go here (e.g. openssl, pkg-config)
        MedievalDependencies = with pkgs; [
          clang-tools
          libclang
          llvmPackages.libclang
          pkg-config
          openssl
          cmake
          openblas
          llvmPackages.openmp
          faiss
          uv
          (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
            # select Python packages here
            langchain
            langchain-ollama
            langchain-core
            langgraph
            matplotlib
            pandas
            langchain-classic
            numpy
          ]))
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

        clangMkShell = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; };

      in
      {
        devShells.default = clangMkShell {
          buildInputs = with pkgs; [
            ollama
            faissForRust
          ] ++ MedievalDependencies ++ DevDependencies ++ RustDependencies;


          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            faissForRust
            pkgs.blas
            pkgs.lapack
          ];

          RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
          LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib";
          shellHook = ''
            export PATH="${pkgs.clangStdenv.cc}/bin:$PATH";
          '';
        };
      });
}

