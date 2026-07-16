.PHONY: build small medium large install clean

build:
	cargo build --release --features model-medium

small:
	cargo build --release --features model-small

medium:
	cargo build --release --features model-medium

large:
	cargo build --release --features model-large

install: medium
	install -Dm755 target/release/rust-background-removal ~/.local/bin/rust-background-removal
	@echo "Installed (medium, 84MB) to ~/.local/bin/rust-background-removal"

small-install: small
	install -Dm755 target/release/rust-background-removal ~/.local/bin/rust-background-removal
	@echo "Installed (small, 42MB) to ~/.local/bin/rust-background-removal"

large-install: large
	install -Dm755 target/release/rust-background-removal ~/.local/bin/rust-background-removal
	@echo "Installed (large, 168MB) to ~/.local/bin/rust-background-removal"

clean:
	cargo clean
