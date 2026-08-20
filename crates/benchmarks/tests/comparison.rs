use benchmarks::comparison_engine as engine;
use std::error::Error;
use std::path::Path;

// Expected file structure:
// test_name
//  |
//  |- config.toml
//  |- harness.py
//  |- harness_rs/
//  |   |
//  |   |- src/
//  |   |- Cargo.toml
//  |
//  |- res/
//      |-
//      |-
// A directory in tests/comparison_tests/ only becomes a test if it contains
// config.toml, harness.py, and a harness_rs/ directory.

fn comparison_test(dir: &str) -> Result<(), Box<dyn Error>> {
    eprintln!("Test Comparison Main for {dir}");

    // python harness

    eprintln!("Python Main for {dir}");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let harness = root
        .join("tests/comparison_tests")
        .join(dir)
        .join("harness_py/");

    engine::run_uv_project(harness.to_str().unwrap())?;

    // cargo harness
    eprintln!("Rust Main for {dir}");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let harness = root
        .join("tests/comparison_tests")
        .join(dir)
        .join("harness_rs/Cargo.toml");

    engine::run_cargo_project(&["--manifest-path", harness.to_str().unwrap()])?;

    Ok(())
}

include!(concat!(env!("OUT_DIR"), "/comparison_test.rs"));
