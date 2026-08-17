use anyhow::Result;
use anyhow::bail;
use std::process::Command;

pub fn run_cargo_project(args: &[&str]) -> Result<()> {
    // let output = Command::new(env!("CARGO_BIN_EXE_your_binary_name"))
    let output = Command::new("cargo").arg("run").args(args).output()?;

    if !output.status.success() {
        bail!("Exit with failure: {}", output.status);
    }

    eprintln!(
        "Stdout: {}\n\nStderr: {}",
        String::from_utf8(output.stdout)?,
        String::from_utf8(output.stderr)?
    );

    Ok(())
}

pub fn run_uv_project(path_to_uv_project: &str) -> Result<()> {
    let out = Command::new("uv")
        .arg("run")
        .arg(path_to_uv_project)
        .output()?;

    if !out.status.success() {
        bail!("Exit with failure: {}", out.status);
    }

    eprintln!(
        "Stdout: {}\n\nStderr: {}",
        String::from_utf8(out.stdout)?,
        String::from_utf8(out.stderr)?
    );
    Ok(())
}
