pub mod comparison_engine;
pub mod test_config;

#[cfg(test)]
mod tests {

    const MIN_PYTHON_VERSION: &str = "3.10";
    const MIN_UV_VERSION: &str = "0.11";

    use {
        anyhow::{Error, Result, bail},
        colored_text::Colorize,
        std::process::Command,
        version_compare::{Cmp, Version, compare, compare_to},
    };

    // Special testing functions

    /// takes in test command, which is written to produce an output with the version
    /// fails if the test command fails
    fn test_cmd_present(mut cmd: Command) -> Result<()> {
        let output = cmd.output()?;

        let stderr = String::from_utf8(output.stderr)?;

        if !stderr.is_empty() {
            dbg!("Standard Error non-empty: {}", stderr);
        }

        if !output.status.success() {
            bail!("Command did not exit successfully")
        }

        Ok(())
    }

    fn check_version<F>(
        mut version_cmd: Command,
        get_version_from_stdout: F,
        msv: Version,
    ) -> Result<()>
    where
        F: for<'a> Fn(&'a [u8]) -> Version<'a>,
    {
        let output = version_cmd.output()?;

        let exec_version = get_version_from_stdout(&output.stdout);

        if exec_version < msv {
            bail!(
                "{} returned an older version {} than the minimum supported version {}",
                version_cmd.get_program().to_str().unwrap().blue(),
                exec_version.red().bold(),
                msv.green().bold()
            )
        }

        Ok(())
    }

    #[test]
    fn assert_python3_present() -> Result<()> {
        let mut cmd = Command::new("python3");
        cmd.arg("--version");

        test_cmd_present(cmd)
    }

    #[test]
    fn assert_uv_present() -> Result<()> {
        let mut cmd = Command::new("uv");
        cmd.arg("--version");

        test_cmd_present(cmd)
    }

    #[test]
    fn pin_python() -> Result<()> {
        let mut cmd = Command::new("python3");
        cmd.arg("--version");

        // make sure python3 is pinned right
        check_version(
            cmd,
            |stdout: &[u8]| {
                let text = std::str::from_utf8(stdout).expect("stdout not valid UTF-8");
                // e.g. "Python 3.11.4\n" -> "3.11.4"
                let raw = text.trim().trim_start_matches("Python ").trim();
                Version::from(raw).expect("failed to parse version")
            },
            Version::from(MIN_PYTHON_VERSION).unwrap(),
        )?;

        Ok(())
    }

    #[test]
    fn pin_uv() -> Result<()> {
        let mut cmd = Command::new("uv");
        cmd.arg("--version");

        // make sure uv is pinned right (>v0.11)
        check_version(
            cmd,
            |stdout: &[u8]| {
                let text = std::str::from_utf8(stdout).expect("stdout not valid UTF-8");
                // e.g. "Python 3.11.4\n" -> "3.11.4"
                let raw = text.trim().trim_start_matches("uv ").trim();
                Version::from(raw).expect("failed to parse version")
            },
            Version::from(MIN_UV_VERSION).unwrap(),
        )?;

        Ok(())
    }
}
