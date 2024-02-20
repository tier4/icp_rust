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
