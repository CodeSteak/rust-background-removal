.PHONY: build small medium large install clean

build:
	cargo build --release --features model-medium

small:
	cargo build --release --features model-small

medium:
	cargo build --release --features model-medium

large:
	cargo build --release --features model-large

install: large
	install -Dm755 target/release/rust-background-removal ~/.local/bin/rust-background-removal
	@echo "Installed to ~/.local/bin/rust-background-removal"

clean:
	cargo clean
