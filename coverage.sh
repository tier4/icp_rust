#!/bin/bash

name="test"
object_option=$( \
  for file in \
    $( \
      RUSTFLAGS="-C instrument-coverage" \
        cargo test --tests --no-run --message-format=json \
          | jq -r "select(.profile.test == true) | .filenames[]" \
          | grep -v dSYM - \
    ); \
  do \
    printf "%s %s " -object $file; \
  done \
)

function cleanup() {
    rm -rf *.profraw *.profdata
}

cleanup && \
    rm -rf coverage && \
    RUSTFLAGS="-C instrument-coverage" cargo test --tests && \
    llvm-profdata-18 merge -sparse *.profraw -o $name.profdata && \
    llvm-cov-18 report $object_option \
        --use-color \
        --ignore-filename-regex='/.cargo/registry' \
        --ignore-filename-regex='/rustc' \
        --summary-only  \
        --instr-profile=$name.profdata && \
    llvm-cov-18 show $object_option \
        --use-color \
        --ignore-filename-regex='/.cargo/registry' \
        --ignore-filename-regex='/rustc' \
        --instr-profile=$name.profdata \
        --show-instantiations --show-line-counts-or-regions \
        --Xdemangler=rustfilt \
        --format=html \
        --output-dir=coverage &&
        cleanup
