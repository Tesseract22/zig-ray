
# I am using zig(https://ziglang.org/). You might want to download the compiler first
build:
	@echo "zig build -Doptimize=ReleaseFast"
	@if zig build -Doptimize=ReleaseFast; then \
		echo "output to zig-out/bin/mp1"; \
	else \
		echo "I am using Zig(https://ziglang.org/). You might want to download the compiler first"; \
	fi
run: build
	zig-out/bin/raytrace $(file)
