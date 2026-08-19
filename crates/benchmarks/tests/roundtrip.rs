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

fn roundtrip_test(dir: &str) -> Result<(), Box<dyn Error>> {
    eprintln!("Roundrip Test Main for {dir}");

    // cargo harness
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let harness = root
        .join("tests/comparison_tests")
        .join(dir)
        .join("harness_rs/Cargo.toml");

    engine::run_cargo_project(&["--manifest-path", harness.to_str().unwrap()])?;

    // FIXME: find a way to test for autheticity

    Ok(())
}

include!(concat!(env!("OUT_DIR"), "/roundtrip_test.rs"));
