

pub fn getTypeLegnth(comptime a: type) comptime_int {
    return switch (@typeInfo(a)) {
            .Struct => |struct_info| struct_info.fields.len,
            .Array, => |array_info| array_info.len,
            .Vector => |vector_info| vector_info.len,
            else => {
                @compileError("Interpolation must be of `struct`, `Array`, or `Vector` type.");
            },
    };
}

pub fn getVectorFromAny(a: anytype) @Vector(getTypeLegnth(@TypeOf(a)), f32)  {
    const N = getTypeLegnth(@TypeOf(a));
    var v: @Vector(N, f32) = undefined;
    switch (@typeInfo(@TypeOf(a))) {
        .Struct => |info| {
            inline for (info.fields, &v) |field, *vi| {
                vi.* = @field(a, field.name);
            }
        },
        .Array,  => |_| {
            inline for (a, 0..) |ai, i| {
                v[i] = ai;
            }
        },
        .Vector,  => |_| {
            v = a;
        },
        else => @compileError("only `struct`, `Array`, and `Vector` is supported"),
    }
    return v;
}

pub fn perspectiveTransform(far: f32, near: f32, a: anytype) ?@TypeOf(a) {
    var v = a;
    v[3] = v[2];
    v[2] = (far+near) / (far-near) * v[2] + (-2*near*far) / (far-near);
    const w = v[3];
    // clipping
    if (v[0] > w or v[0] < -w or v[1] > w or v[1] < -w or v[2] > w or v[2] < -w) return null;
    // divide everything by w, except w => 1/w instead of 1
    v /= @splat(w);
    v[3] = 1/w;
    return v;
}

pub fn vecDescartesLen(v: anytype) f32 {
    return @sqrt(@reduce(.Add, v * v));
}

pub fn dot(a: anytype, b: @TypeOf(a)) @typeInfo(@TypeOf(a)).Vector.child {
    return @reduce(.Add, a * b);
}

pub fn normalize(v: anytype) @TypeOf(v) {
    if (@typeInfo(@TypeOf(v)) == .Vector) {
        return v / @as(@TypeOf(v) ,@splat(vecDescartesLen(v)));
    }
    var sum: f32 = 0;
    for (v) |vi| {
        sum += (vi * vi);
    }
    sum = @sqrt(sum);
    if (sum == 0) return v;
    var res = v;
    for (res, 0..) |_, i| {
        res[i] /= sum;
    }
    return res;
}


pub fn rgbaU32(r: f32, g: f32, b: f32, a: f32) u32 {
    return rgbaArrayToU32(.{ r, g, b, a });
}
pub fn rgbaArrayToU32(arr: [4]f32) u32 {
    var res: u32 = 0;
    for (arr, 0..) |color, i| {
        const c = @as(u32, @intFromFloat(color * 255));
        res += @as(u32, @intCast(c)) << (@as(u5, @intCast(i * 8)));
    }
    return res;
}

pub fn rgbaVecToU32(arr: @Vector(4, f32)) u32 {
    var v: @Vector(4, u32) = @intFromFloat(@as(@Vector(4, f32),@splat(255)) * arr);
    v = @min(@as(@Vector(4, u32), @splat(255)),  v);
    v *= @Vector(4, u32) {1, 1 << 8, 1 << 16, 1 << 24};
    return @reduce(.Add, v);
}

pub fn rgbToGreyScale(r: f32, g: f32, b: f32) f32 {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}
pub fn color4ToGreyScale(c: Color4) f32 {
    const f = c.toF32();
    return rgbToGreyScale(f[0], f[1], f[2]);
}

pub const Color4 = packed struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,

    pub fn toU32(self: Color4) *u32 {
        return @constCast(@ptrCast(&self));
    }
    pub fn fromU32(n: u32) *Color4 {
        return @constCast(@ptrCast(&n));
    }

    // pub fn toVec4(self: Color4) *@Vector(N, 4) {

    // }

    pub fn toF32(self: Color4) [4]f32 {
        return .{ 
            @as(f32,@floatFromInt(self.r))/255,
            @as(f32,@floatFromInt(self.g))/255,
            @as(f32,@floatFromInt(self.b))/255,
            @as(f32,@floatFromInt(self.a))/255,};
    }
};

pub fn crossProduct(v1: anytype, v2: @TypeOf(v1)) @TypeOf(v1) {
    return .{
        v1[1] * v2[2] - v1[2] * v2[1], -(v1[0] * v2[2] - v1[2] * v2[0]),
            v1[0] * v2[1] - v1[1] * v2[0]
    };
}
pub fn greaterThan(v: anytype, s: f32) bool {
    return @reduce(.Or, v > @as(@TypeOf(v), @splat(s)));
}

pub fn scalarVecMul(s: f32, v: anytype) @TypeOf(v) {
    return @as(@TypeOf(v), @splat(s)) * v;
}



