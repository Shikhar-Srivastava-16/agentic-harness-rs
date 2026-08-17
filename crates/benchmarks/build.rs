use std::env::var;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use walkdir::DirEntry;
use walkdir::WalkDir;

extern crate version_check as rustc;

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tests/comparison_tests");

    // version and date checks and safeties
    if rustc::is_min_version("1.96.1").unwrap_or(false) {
        println!("cargo:rustc-cfg=question_mark_operator");
    } else {
        println!(
            "cargo::warning=rustc is pinned to older version that recommended; this may cause compilation problems"
        )
    }

    match rustc::is_min_date("2026-06-01") {
        Some(true) => {}
        Some(false) => println!(
            "cargo::warning=rustc is pinned to older date than recommended; this may cause compilation problems"
        ),
        None => println!("cargo::warning=could not determine the rustc version"),
    };

    write_comparison_tests()?;

    Ok(())
}

fn write_comparison_tests() -> std::io::Result<()> {
    let out_dir = var("OUT_DIR").map_err(std::io::Error::other)?; // wrapping in a std::io::Error to match main's error type
    let manifest_dir = var("CARGO_MANIFEST_DIR").map_err(std::io::Error::other)?;
    let mut f = File::create(format!("{}/comparison_test.rs", out_dir))?;
    let base_dir = Path::new(&manifest_dir);

    // Entry needs to have been ensured to be Ok
    for entry in comparison_test_dirs(&manifest_dir)? {
        write!(
            f,
            include_str!("./tests/comparison_template"),
            test_name = entry
                .path()
                .strip_prefix(base_dir)
                .unwrap()
                .to_str()
                .unwrap()
                .replace(['-', '/'], "_"),
            test_dir = entry.path().to_str().unwrap(),
            ignore_attr = "",
        )?;
    }

    Ok(())
}

/// Look for the directories which qualify as 'test directories' and can be used to run tests.
/// This will recursively search the entire directory
/// tests can be inside other tests
fn comparison_test_dirs(manifest_dir: &str) -> std::io::Result<Vec<DirEntry>> {
    let root = Path::new(manifest_dir).join("tests/comparison_tests");
    let mut dirs = Vec::new();

    for entry in WalkDir::new(&root) {
        let entry = entry?;
        if !entry.file_type().is_dir() {
            continue;
        }

        let path = entry.path();
        if path.join("config.toml").is_file()
            && path.join("harness_py").is_dir()
            && path.join("harness_rs").is_dir()
        {
            dirs.push(entry);
        }
    }

    Ok(dirs)
}
