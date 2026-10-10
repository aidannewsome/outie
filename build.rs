//! Embree is linked from EMBREE_DIR when set, else from ./embree, where .github/build-embree.sh builds it as static
//! libraries for the wheels to carry, else from where Homebrew and Linux packages install it, as a shared library.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=EMBREE_DIR");
    println!("cargo:rerun-if-changed=embree/lib");
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("embree");
    let dir = std::env::var("EMBREE_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| here.exists().then_some(here));
    match dir {
        Some(dir) if find_static(&dir.join("lib")).is_some() => link_static(&dir.join("lib")),
        Some(dir) => {
            for lib in ["lib", "lib64"] {
                println!("cargo:rustc-link-search=native={}", dir.join(lib).display());
            }
            println!("cargo:rustc-link-lib=dylib=embree4");
        }
        None => {
            for dir in [
                "/opt/homebrew/lib",
                "/usr/local/lib",
                "/usr/lib",
                "/usr/lib64",
            ] {
                println!("cargo:rustc-link-search=native={dir}");
            }
            println!("cargo:rustc-link-lib=dylib=embree4");
        }
    }
}

/// The static libraries Embree installs in a folder, by name, if it holds its main one.
fn find_static(lib: &Path) -> Option<Vec<String>> {
    let names: Vec<String> = std::fs::read_dir(lib)
        .ok()?
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let stem = name
                .strip_suffix(".a")
                .map(|s| s.trim_start_matches("lib"))
                .or_else(|| name.strip_suffix(".lib"))?;
            Some(stem.to_string())
        })
        .collect();
    names.iter().any(|n| n == "embree4").then_some(names)
}

/// Embree and its parts, in the order its CMake targets link them: the main library, each instruction set's, then
/// the helpers; then the C++ runtime, which a static Embree needs from the system.
fn link_static(lib: &Path) {
    let names = find_static(lib).unwrap();
    println!("cargo:rustc-link-search=native={}", lib.display());
    let isas = names.iter().filter(|n| n.starts_with("embree_")).cloned();
    let helpers = ["sys", "math", "simd", "lexers", "tasking"]
        .map(String::from)
        .into_iter()
        .filter(|h| names.contains(h));
    for name in std::iter::once("embree4".to_string())
        .chain(isas)
        .chain(helpers)
    {
        println!("cargo:rustc-link-lib=static={name}");
    }
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
        "windows" => {}
        _ => println!("cargo:rustc-link-arg=-lstdc++"), // after Embree on the line, or the linker drops it as unneeded
    }
}
