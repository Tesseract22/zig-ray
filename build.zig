const std = @import("std");
// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {

    const target = b.standardTargetOptions(.{});


    const optimize = b.standardOptimizeOption(.{});
    // cli app
    // const cli_exe = b.addExecutable(.{
    //     .name = "raytrace",
    //     // In this case the main source file is merely a path, however, in more
    //     // complicated build scripts, this could be a generated file.
    //     .root_source_file = .{ .path = "src/main.zig" },
    //     .target = target,
    //     .optimize = optimize,
    // });
    // cli_exe.linkLibC();
    // // exe.linkLibCpp();
    // cli_exe.addIncludePath(.{ .path = "lodepng/" });
    const loadepng_flags = &[_][]const u8{
        "-ansi",
        "-O3",
    };
    // cli_exe.addCSourceFile(.{ .file = .{ .path = "lodepng/lodepng.c" }, .flags = loadepng_flags });
    // b.installArtifact(cli_exe);


    // gui app where we can see the rendering process
    const gui_exe = b.addExecutable(.{
        .name = "main",

        .root_source_file = .{ .path = "src/gui.zig" },
        .target = target,
        .optimize = optimize,
    });
    gui_exe.linkLibC();
    gui_exe.addIncludePath(.{ .path = "lodepng/" });
    gui_exe.addCSourceFile(.{ .file = .{ .path = "lodepng/lodepng.c" }, .flags = loadepng_flags });

    // optionally enabling raylib gui with -Dgui
    var build_opt = b.addOptions();
    const gui_enabled = b.option(bool, "gui", "whether to integrate with raylib") orelse false;
    build_opt.addOption(bool, "gui", gui_enabled);
    gui_exe.root_module.addOptions("config", build_opt);

    if (gui_enabled)  {
        const raySdk = @import("raylib/src/build.zig");
        const raylib = raySdk.addRaylib(b, target, optimize, .{}) catch unreachable;
        gui_exe.addIncludePath(.{ .path = "raylib/src" });
        gui_exe.linkLibrary(raylib);
    }

    b.installArtifact(gui_exe);



    // const run_cmd = b.addRunArtifact(cli_exe);

    // run_cmd.step.dependOn(b.getInstallStep());

    // if (b.args) |args| {
    //     run_cmd.addArgs(args);
    // }


    // const run_step = b.step("run", "Run the app");
    // run_step.dependOn(&run_cmd.step);


}
