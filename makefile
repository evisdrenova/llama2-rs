# Rust Makefile for LLaMA implementation
# Uses cargo for building and running

# Default target
.PHONY: all
all: build

# Basic optimized build (equivalent to -O3)
.PHONY: build
build:
	cargo build --release

# Debug build (equivalent to -g)
.PHONY: debug
debug:
	cargo build

# Run the main binary
.PHONY: run
run: build
	cargo run --release

# Fast optimized build with native CPU features
.PHONY: fast
fast:
	RUSTFLAGS="-C target-cpu=native" cargo build --release

# Build with parallel processing optimizations
.PHONY: parallel
parallel:
	RUSTFLAGS="-C target-cpu=native" cargo build --release --features parallel

# Cross-compile for Windows (requires windows target)
.PHONY: win64
win64:
	cargo build --release --target x86_64-pc-windows-gnu

# Profile-guided optimization build
.PHONY: pgo
pgo:
	RUSTFLAGS="-C target-cpu=native -C codegen-units=1" cargo build --release

# Build with link-time optimization
.PHONY: lto
lto:
	RUSTFLAGS="-C target-cpu=native -C lto=fat" cargo build --release

# Ultra-optimized build (combines multiple optimizations)
.PHONY: ultra
ultra:
	RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1" cargo build --release

# Run tests
.PHONY: test
test:
	cargo test

# Run tests with output
.PHONY: test-verbose
test-verbose:
	cargo test -- --nocapture

# Run benchmarks (if you have criterion benchmarks)
.PHONY: bench
bench:
	cargo bench

# Check code without building
.PHONY: check
check:
	cargo check

# Format code
.PHONY: fmt
fmt:
	cargo fmt

# Lint code
.PHONY: clippy
clippy:
	cargo clippy -- -D warnings

# Clean build artifacts
.PHONY: clean
clean:
	cargo clean

# Install binary to system
.PHONY: install
install: build
	cargo install --path .

# Generate documentation