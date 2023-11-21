
# I am using zig(https://ziglang.org/). You might want to download the compiler first
build:
	zig build -Doptimize=ReleaseFast
run: build
	zig-out/bin/raytrace $(file)
