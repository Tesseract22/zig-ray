const std = @import("std");
const simd = std.simd;
pub const util = @import("util.zig");
const DepthBufType = f32;
const Canvas = @This();

// x,y,z,w,r,g,b,a,s,t
const X = 0;
const Y = 1;
const Z = 2;
const W = 3;
const R = 4;
const G = 5;
const B = 6;
const A = 7;
const S = 8;
const T = 9;


pub fn DDAIterator(comptime N: comptime_int) type {
    const VecN = @Vector(N, f32);
    return struct {
        p: VecN,
        s: VecN,
        bd: f32,
        axis: u8,
        pub fn next(self: *@This()) ?VecN {
            if (self.p[self.axis] < self.bd) {
                defer self.p += self.s;
                return self.p;
            }
            return null;
        }
    };
    
}

width: u32,
height: u32,
data: []util.Color4,
depth: ?[]DepthBufType,
pub fn init(width: u32, height: u32, data: []util.Color4, depth: ?[]DepthBufType) !Canvas {
    std.debug.assert(data.len == width * height);
    if (depth) |d| std.debug.assert(d.len == width * height);
    return .{ .height = height, .width = width, .data = data, .depth = depth };
}

pub fn resetDepth(self: *Canvas) void {
    if (self.depth) |d| @memset(d, std.math.floatMin(f32));
}

pub fn resize(self: *Canvas, width: u32, height: u32, data: []u8) !void {
    self.data = data;
    self.height = height;
    self.width = width;
}
pub fn deinit(self: *Canvas) void {
    self.allocator.free(self.data);
}
pub fn set(self: *Canvas, pos: [3]f32, el: u32) void {
    if (pos[0] < 0 or pos[0] >= @as(f32, @floatFromInt(self.width)) or pos[1] < 0 or pos[1] >= @as(f32, @floatFromInt(self.height))) return;
    const x: u32 = @intFromFloat(pos[0]);
    const y: u32 = @intFromFloat(pos[1]);
    if (self.depth) |d| {
        if (d[y * self.width + x] >= pos[2]) {
            return;
        }
        d[y * self.width + x] = pos[2];
    }

    self.data[y * self.width + x] = util.Color4.fromU32(el).*;
}

pub fn setRaw(self: *Canvas, pos: [2]u32, el: u32) void {
    self.data[pos[1] * self.width + pos[0]] = util.Color4.fromU32(el).*;
}
pub fn viewPortTransform2(self: *Canvas, pos: anytype) void {
    const w: f32 = @floatFromInt(self.width);
    const h: f32 = @floatFromInt(self.height);
    var res = pos;
    res[0] = (pos[0] + 1) * w / 2;
    res[1] = (pos[1] + 1) * h / 2;
}
pub fn ShaderFn(comptime N: comptime_int) type {
    return fn (cv: *Canvas, v: @Vector(N, f32)) u32;
}       
fn drawTriangleN(self: *Canvas, comptime N: comptime_int, v1: @Vector(N, f32), v2: @Vector(N, f32), v3: @Vector(N, f32), shader: ShaderFn(N)) void {
    const VecN = @Vector(N, f32);
    var a = &v1;
    var b = &v2;
    var c = &v3;
    if (a[1] > c[1]) std.mem.swap(VecN, @constCast(a), @constCast(c));
    if (a[1] > b[1]) std.mem.swap(VecN, @constCast(a), @constCast(b));
    if (b[1] > c[1]) std.mem.swap(VecN, @constCast(b), @constCast(c));

    var top_to_bot = DDA(N, a.*, c.*, 1) orelse {
        return;
    };
    var pos1: VecN = undefined;
    var pos2: VecN = undefined;
    if (DDA(N, a.*, b.*, 1)) |*top_to_mid| {
        while (true) {
            pos1 = @constCast(top_to_mid).next() orelse break;
            pos2 = top_to_bot.next() orelse break;
            var x_dda = DDA(N, pos1, pos2, 0) orelse continue;
            while (x_dda.next()) |pos| {
                const pos_alt = pos / @as(VecN, @splat(pos[W]));
                self.set(.{ pos[0], pos[1], pos[2] }, shader(self, pos_alt));
            }
        }
    }
    var mid_to_bot = DDA(N, b.*, c.*, 1) orelse return;
    while (true) {
        pos1 = mid_to_bot.next() orelse break;
        pos2 = top_to_bot.next() orelse break;
        var x_dda = DDA(N, pos1, pos2, 0) orelse continue;
        while (x_dda.next()) |pos| {
            const pos_alt = pos / @as(VecN, @splat(pos[W]));
            self.set(.{ pos[0], pos[1], pos[2] }, shader(self, pos_alt));
        }
    }
    
}
const getTypeLegnth = util.getTypeLegnth;
pub fn drawTriangleAny(self: *Canvas, a: anytype, b: anytype, c: anytype, shader: ShaderFn(getTypeLegnth(@TypeOf(a)))) void {
    const N = comptime getTypeLegnth(@TypeOf(a));
    comptime {
        if (N < 4) {
            @compileError(std.fmt.comptimePrint("Expect point length >= 4, got {}", .{N}));
        }
        if (N != getTypeLegnth(@TypeOf(b))) {
            @compileError(std.fmt.comptimePrint("length of a ({}) != length of b ({})", .{N,  getTypeLegnth(@TypeOf(b))}));
        }
        if (N != getTypeLegnth(@TypeOf(c))) @compileError("length of a != length of c");
    }


    var vecs: [3]@Vector(N, f32) = undefined;
    vecs[0] = util.getVectorFromAny(a);
    vecs[1] = util.getVectorFromAny(b);
    vecs[2] = util.getVectorFromAny(c);
    
    for (&vecs) |*v| {
        self.viewPortTransform2(v);
    }
    self.drawTriangleN(N, vecs[0], vecs[1], vecs[2], shader);
    
    
}
pub fn fillU32(self: *Canvas, el: u32) void {
    
    @memset(self.data, util.Color4.fromU32(el).*);
}
// pub fn DDA(a: [2]f32, b: [2]f32, axis: u8) ?DDAIterator {
//     if (a[axis] == b[axis]) return null;
//     if (a[axis] > b[axis]) return DDA(b, a, axis);
//     const delta: [2]f32 = .{ b[0] - a[0], b[1] - a[1] };
//     const s: [2]f32 = .{ delta[0] / delta[axis], delta[1] / delta[axis] };
//     const e = @ceil(a[axis]) - a[axis];
//     const p: [2]f32 = .{ a[0] + e * s[0], a[1] + e * s[1] };
//     return .{ .p = p, .s = s, .axis = axis, .bd = b[axis] };
// }
pub fn DDA(comptime N: comptime_int, a: @Vector(N, f32), b: @Vector(N, f32), axis: u8) ?DDAIterator(N) {
    const VecN = @Vector(N, f32);
    if (a[axis] == b[axis]) return null;
    if (a[axis] > b[axis]) return DDA(N, b, a, axis);
    const delta = b - a;
    const s = delta / @as(VecN, @splat(delta[axis]));
    const e = @ceil(a[axis]) - a[axis];
    const p = a + @as(VecN, @splat(e)) * s;
    return .{ .p = p, .s = s, .axis = axis, .bd = b[axis] };
}
