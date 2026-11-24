use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FLINT_ENGINE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=../engine/include/engine_api.h");
    if env::var_os("CARGO_FEATURE_LINK_ENGINE").is_none() {
        return;
    }

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let build =
        root.join(env::var_os("FLINT_ENGINE_BUILD_DIR").unwrap_or("../engine/build".into()));
    println!(
        "cargo:rerun-if-changed={}",
        build.join("libengine.a").display()
    );
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=static=engine");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    let cuda = PathBuf::from(env::var_os("CUDA_HOME").unwrap_or("/usr/local/cuda".into()));
    let lib = ["lib64", "targets/x86_64-linux/lib", "lib"]
        .iter()
        .map(|dir| cuda.join(dir))
        .find(|dir| dir.join("libcudart.so").is_file())
        .unwrap_or_else(|| cuda.join("lib64"));
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
