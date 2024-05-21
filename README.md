# ICP (Iterative Closest Point) in Rust

![](https://github.com/tier4/icp_rust/actions/workflows/test.yml/badge.svg)

![](assets/icp.gif)

## Target environment

Linux x86_64

## How to run

```
$ rustup toolchain install nightly-x86_64-unknown-linux-gnu
$ cargo run --example scan2d --release   # 2D scan example
$ cargo run --example scan3d --release   # 3D scan example
```

## Debug

Since icp_rust is implemented for no_std environment, you may need to explicitly pass `--features std` as an argument e.g. `cargo test --features std` when debugging with `println`. 

## Docs

You can generate the API documentation with `cargo doc --no-deps`. The detailed API doc has not been created yet, though.
