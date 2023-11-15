const Canvas = @import("canvas.zig");
const util = @import("util.zig");
const scalarVec = util.scalarVecMul;
const std = @import("std");

const lodepng = @cImport({
    @cInclude("lodepng.h");
});

const Vec3 = @Vector(3, f32);
const Vec4 = @Vector(4, f32);
const t_min = 0.01; // avoid self-intersecting
const secondary_ray_n = 2;
const secondary_ray_off = 0.05;
var RandGen = std.rand.DefaultPrng.init(0);
const KeyWord = enum(u32) {
    png,
    sphere,
    plane,
    tri,
    xyz,
    sun,
    color,
    texture,
    texcoord,
    expose,
    shininess,
    roughness,
    transparency,
    ior,
    bounces,
    up,
    eye,
    forward,
    fisheye,
    panorama,
    dof,
    bulb,
    aa,
    gi,
    const MatchResult = struct {
        keyword: KeyWord,
        remain: []const u8,
    };
    pub fn match(s: []const u8) ?MatchResult {
        const type_info = @typeInfo(KeyWord);
        inline for (type_info.Enum.fields) |f| {
            if (startsWith(s, f.name)) {
                return .{.keyword = @enumFromInt(f.value), .remain = s[f.name.len..]};
            }
        }
        return null;
    }
};
pub fn parseCommands(path: []const u8, allocator: std.mem.Allocator, data: *Data) !void {
    var input = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer input.close();
    var buffer_reader = std.io.bufferedReader(input.reader());
    
    while (try buffer_reader.reader().readUntilDelimiterOrEofAlloc(allocator, '\n', 1024 * 1024)) |line| {
        // std.debug.print("{s}\n", .{line});
        defer allocator.free(line);
        const match_res = KeyWord.match(line) orelse continue;
        var it = splitSpace(match_res.remain);
        switch (match_res.keyword) {
            .gi => {
                State.gi = parseU32(it.next().?);
            },
            .png => {
                State.width = parseU32(it.next().?);
                State.height = parseU32(it.next().?);
                const out = it.next().?;
                const duped = allocator.allocSentinel(u8, out.len, 0) catch unreachable;
                @memset(duped, 0);
                @memcpy(duped, out);
                State.output_path = duped;
            },
            .sphere => {
                var sphere: Sphere = undefined;
                sphere.c = .{
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                };
                sphere.r = parseF32(it.next().?);
                if (State.texture_active) {
                    sphere.color = .{.texture = State.texture};
                } else {
                    sphere.color = .{.color = State.color};
                }
                sphere.material = State.material.actual();
                sphere.aabb.min = @splat(std.math.floatMin(f32));
                sphere.aabb.max = @splat(std.math.floatMax(f32));
                sphere.aabb.min = sphere.c - @as(Vec3, @splat(sphere.r));
                sphere.aabb.max = sphere.c + @as(Vec3, @splat(sphere.r));
                data.geoms.append(.{.sphere = sphere}) catch unreachable;
            },
            .plane => {
                var plane: Plane = undefined;
                plane.n = .{
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                };
                const d = parseF32(it.next().?);
                plane.p = @as(Vec3, @splat(-d)) * plane.n / @as(Vec3, @splat(util.vecDescartesLen(plane.n)));
                plane.n = util.normalize(plane.n);
                plane.color = State.color;
                plane.material = State.material.actual();
                data.planes.append(plane) catch unreachable;
            },
            .tri => {
                var tri: Triangle = undefined;
                const vertexes = data.vertexes.items;
                for (0..3) |i| {
                    var index = parseI32(it.next().?);
                    index = if (index > 0) index - 1 else index + @as(i32, @intCast(vertexes.len));
                    tri.vs[i] = vertexes[@intCast(index)];
                }
                tri.n = util.normalize(util.crossProduct(tri.vs[0].v - tri.vs[1].v, tri.vs[1].v - tri.vs[2].v));
                tri.e1 = util.crossProduct(tri.vs[2].v - tri.vs[0].v, tri.n);
                tri.e1 /= @splat(util.dot(tri.e1, tri.vs[1].v - tri.vs[0].v));
                tri.e2 = util.crossProduct(tri.vs[1].v - tri.vs[0].v, tri.n);
                tri.e2 /= @splat(util.dot(tri.e2, tri.vs[2].v - tri.vs[0].v));
                tri.color = if (State.texture_active) .{.texture = State.texture } else .{.color = State.color};
                tri.material = State.material.actual();

                tri.aabb =  .{.min = tri.vs[0].v, .max = tri.vs[0].v };
                for (1..3) |i| {
                    tri.aabb.min = @min(tri.aabb.min, tri.vs[i].v);
                    tri.aabb.max = @max(tri.aabb.max, tri.vs[i].v);
                }
                data.geoms.append(.{.triangle = tri}) catch unreachable;
            },
            .xyz => data.vertexes.append(.{
                    .v = .{ parseF32(it.next().?), parseF32(it.next().?), parseF32(it.next().?),},
                    .texcoord = State.texcoord,
                }
            ) catch unreachable,
            .sun => {
                var sun: LightSource = undefined;
                sun.dir = util.normalize(Vec3 {
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                });
                sun.color = State.color;
                sun.is_sun = true;
                data.lights.append(sun) catch unreachable;
            },
            .bulb => {
                var bulb: LightSource = undefined;
                bulb.dir = Vec3 {
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                    parseF32(it.next().?),
                };
                bulb.color = State.color;
                bulb.is_sun = false;
                data.lights.append(bulb) catch unreachable;
            },
            .color => {
                for (0..3) |i| {
                    State.color[i] = parseF32(it.next().?);
                }
                State.texture_active = false;
            },
            .texture => {
                const text_path = it.next().?;
                if (std.mem.eql(u8, text_path, "none")) {
                    State.texture_active = false;
                    continue;
                }
                const c_text_path = allocator.allocSentinel(u8, text_path.len, 0) catch unreachable;
                defer allocator.free(c_text_path);
                @memcpy(c_text_path, text_path);
                const err = lodepng.lodepng_decode32_file(
                        @ptrCast(&State.texture.data), 
                            @ptrCast(&State.texture.w), 
                            @ptrCast(&State.texture.h), c_text_path);
                if (err != 0) {
                    std.debug.print("load texture failed: [{}] {s}\n", .{err, lodepng.lodepng_error_text(err)});
                    unreachable;
                }

                State.texture_active = true;
            },
            .texcoord => {
                State.texcoord = .{parseF32(it.next().?), parseF32(it.next().?)};
            },
            .expose => State.expose = parseF32(it.next().?),
            .shininess => {
                State.material.shininess[0] = parseF32(it.next().?);
                if (it.next()) |s1| {
                    State.material.shininess[1] = parseF32(s1);
                    State.material.shininess[2] = parseF32(it.next().?);
                } else {
                    State.material.shininess[1] = State.material.shininess[0];
                    State.material.shininess[2] = State.material.shininess[0];
                }
            },
            .transparency => {
                State.material.transparency[0] = parseF32(it.next().?);
                if (it.next()) |s1| {
                    State.material.transparency[1] = parseF32(s1);
                    State.material.transparency[2] = parseF32(it.next().?);
                } else {
                    State.material.transparency[1] = State.material.transparency[0];
                    State.material.transparency[2] = State.material.transparency[0];
                }
            },
            .ior => {
                State.material.index = parseF32(it.next().?);
            },
            .roughness => {
                State.material.roughness = parseF32(it.next().?);
            },
            .bounces => {
                State.bounces = parseU32(it.next().?);
            },
            .forward => {
                for (0..3) |i| {
                    State.f[i] = parseF32(it.next().?);
                }
                State.r = util.normalize(util.crossProduct(State.f, State.u));
                State.u = util.normalize(util.crossProduct(State.r, State.f));
            },
            .up => {
                for (0..3) |i| {
                    State.u[i] = parseF32(it.next().?);
                }
                State.u = util.normalize(State.u);
                State.r = util.normalize(util.crossProduct(State.f, State.u));
                State.u = util.normalize(util.crossProduct(State.r, State.f));
            },
            .eye => for (0..3) |i| { State.e[i] = parseF32(it.next().?); },
            .fisheye => State.fisheye = true,
            .panorama => State.panorama = true,
            .dof => {
                State.dof = .{parseF32(it.next().?), parseF32(it.next().?)};
            },
            .aa => {
                State.aa = parseU32(it.next().?);
            },
        }
    }
}
fn parseU32(s: []const u8) u32 {
    return std.fmt.parseInt(u32, s, 10) catch unreachable;
}
fn parseI32(s: []const u8) i32 {
    return std.fmt.parseInt(i32, s, 10) catch unreachable;
}
fn parseF32(s: []const u8) f32 {
    return std.fmt.parseFloat(f32, s) catch unreachable;
}



var State: struct {
    gi: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    output_path: [:0]const u8 = "",
    color: Vec3 = .{1,1,1},
    texture: Texture = undefined,
    texcoord: @Vector(2, f32) = .{0,0},
    texture_active: bool = false,
    expose: ?f32 = null,
    material: Material = .{},
    bounces: u32 = 4,
    fisheye: bool = false,
    panorama: bool = false,
    aa: u32 = 0,
    e: Vec3 = .{0,0,0},
    f: Vec3 = .{0,0,-1},
    r: Vec3 = .{1,0,0},
    u: Vec3 = .{0,1,0},
    dof: @Vector(2, f32) = .{0,0},
} = .{};
const Ray = struct {
    origin: Vec3,
    n: Vec3, // normalized
    const Tuple = std.meta.Tuple;
    const Intersection = struct {
        t: f32,
        object: union(enum) {
            sphere: Tuple(&[_]type{*Sphere, bool}),
            plane: *Plane,
            triangle: Tuple(&[_]type{*Triangle, Vec3}),
        }
    };
    pub fn dof(self: Ray) Ray {
        const angle = RandGen.random().float(f32) * std.math.pi * 2;
        const len = RandGen.random().float(f32) * State.dof[1];
        var r = self;
        r.origin += scalarVec(@cos(angle)*len, State.u);
        r.origin += scalarVec(@sin(angle)*len, State.r);
        r.n =  self.origin + scalarVec(State.dof[0], self.n) - r.origin;
        r.n /= @splat(State.dof[0]);
        r.n = util.normalize(r.n);
        return r;
        
    }
    pub fn sphere(self: Ray, s: *Sphere) ?Tuple(&[_]type{f32, bool}) {
        const center_to_eye = s.c - self.origin;
        const r2 = s.r * s.r;
        const is_inside = @reduce(.Add, center_to_eye * center_to_eye) < r2;
        const tc = util.dot(center_to_eye, self.n) / util.vecDescartesLen(self.n);

        if (!is_inside and tc < 0) {
            return null;
        }
        const dv = self.origin + @as(Vec3, @splat(tc)) * self.n - s.c;
        const d2 = @reduce(.Add, dv * dv);
        if (!is_inside and r2 < d2) {
            return null;
        }
        const t_off = @sqrt(r2 - d2) / util.vecDescartesLen(self.n);
        const t = if (is_inside) tc + t_off else tc - t_off;
        return .{t, is_inside};

    }
    pub fn plane(self: Ray, p: *Plane) ?f32 {
        const t = util.dot((p.p - self.origin), p.n) / util.dot(self.n, p.n);
        return if (t > 0) t else null;
    }

    pub fn triangle(self: Ray, tri: *Triangle) ?Tuple(&[_]type{f32, Vec3})  {
        var p = Plane {.n = tri.n, .p = tri.vs[0].v, .color = undefined, .material = undefined };
        if (self.plane(&p)) |t| {
            const point = @as(Vec3, @splat(t)) * self.n + self.origin;
            var bs: Vec3 = undefined;
            bs[1] = util.dot(tri.e1, point - tri.vs[0].v);
            bs[2] = util.dot(tri.e2, point - tri.vs[0].v);
            bs[0] = 1 - bs[1] - bs[2];
            if (@reduce(.Or, bs < @as(Vec3, @splat(0.0)))) return null;

            return .{t, bs};
        }
        return null;
    }
    pub fn intersectLinear(self: Ray, data: Data) ?Intersection {
        var intersection: Intersection = .{.t = std.math.floatMax(f32), .object = undefined };
        for (data.geoms.items) |*geom| {
            switch (geom.*) {
                .sphere => |*s| {
                    const ts = self.sphere(s) orelse continue;
                    if (ts[0] < t_min) continue; 
                    if (ts[0] < intersection.t) {
                        intersection.t = ts[0];
                        intersection.object = .{.sphere = .{s, ts[1]} };
                    }
                },
                .triangle => |*tri| {
                    const ts = self.triangle(tri) orelse continue;
                    if (ts[0] < t_min) continue; 
                    if (ts[0] < intersection.t) {
                        intersection.t = ts[0];
                        intersection.object = .{ .triangle = .{tri, ts[1]} };
                    }
                },
            }
        }
        for (data.planes.items) |*p| {
            const ts = self.plane(p) orelse continue;
            if (ts < t_min) continue; 
            if (ts < intersection.t) {
                intersection.t = ts;
                intersection.object = .{.plane = p };
            }
        }
        return if (intersection.t == std.math.floatMax(f32)) null else intersection;
    }
    pub fn intersect(self: Ray, data: Data) ?Intersection {
        var it: Intersection = .{.t = std.math.floatMax(f32), .object = undefined };
        if (self.intersectBVH(0, data.bvh_tree.items)) |sub_it| {
            if (sub_it.t < it.t) it = sub_it;
        }
        for (data.planes.items) |*p| {
            const ts = self.plane(p) orelse continue;
            if (ts < t_min) continue; 
            if (ts < it.t) {
                it.t = ts;
                it.object = .{.plane = p };
            }
        }
        return if (it.t == std.math.floatMax(f32)) null else it;
    }
    pub fn intersectBVH(self: Ray, root: usize, nodes: []BVHNode) ?Intersection {
        const root_node = nodes[root];
        var it: Intersection = .{.t = std.math.floatMax(f32), .object = undefined };
        var is_leaf = true;
        if (self.intersectAABB(root_node.aabb)) {
            if (root_node.left < nodes.len) {
                is_leaf = false;
                if (self.intersectBVH(root_node.left, nodes)) |sub_it| {
                    
                    if (sub_it.t < it.t) it = sub_it;
                }
            }
            if (root_node.right < nodes.len) {
                is_leaf = false;
                if (self.intersectBVH(root_node.right, nodes)) |sub_it| {
                    if (sub_it.t < it.t) it = sub_it;
                }
            }
            if (is_leaf) {
                for (root_node.geoms) |*geom| {
                    switch (geom.*) {
                        .sphere => |*s| {
                            const ts = self.sphere(s) orelse continue;
                            if (ts[0] < t_min) continue; 
                            if (ts[0] < it.t) {
                                it.t = ts[0];
                                it.object = .{.sphere = .{s, ts[1]} };
                            }
                        },
                        .triangle => |*tri| {
                            const ts = self.triangle(tri) orelse continue;
                            if (ts[0] < t_min) continue; 
                            if (ts[0] < it.t) {
                                it.t = ts[0];
                                it.object = .{ .triangle = .{tri, ts[1]} };
                            }
                        },
                    }
                }
            }

            
        }
        return if (it.t == std.math.floatMax(f32)) null else it;
    }
    pub fn intersectAABB(self: Ray, aabb: AABB) bool {
        const t1 = (aabb.min - self.origin) / self.n;
        const t2 = (aabb.max - self.origin) / self.n;

        const tmin = @min(t1, t2);
        const tmax = @max(t1, t2);

        const min = @reduce(.Max, tmin);
        const max = @reduce(.Min, tmax);
        return max >= min;
    }
    pub fn trace(ray: Ray, data: Data, depth: u32, gi: u32, inside: bool) Vec4 {
        const intersection = ray.intersect(data) orelse return .{0,0,0,0};
        var pixel: Vec4 = .{0,0,0,1};
        const p = @as(Vec3, @splat(intersection.t)) * ray.n + ray.origin;
        var normal = switch (intersection.object) {
            .sphere => |sph| (p - sph[0].c) / @as(Vec3, @splat(sph[0].r)),
            .plane => |pl| pl.n,
            .triangle => |tri| tri[0].n,
        };

        if (util.dot(normal, ray.n) > 0) {
            normal = -normal;
        }
        const color = switch (intersection.object) {
            .sphere => |sph| switch (sph[0].color) {
                .color => |color| color,
                .texture => |texture| blk: {
                    const u = 0.5 + std.math.atan2(f32, normal[0], normal[2]) / (std.math.pi * 2);
                    const v = 0.5 + std.math.asin(-normal[1]) / (std.math.pi);
                    break :blk std.simd.extract(texture.get(u,v), 0, 3);
                }
            },
            .plane => |pl| pl.color,
            .triangle => |tri| switch (tri[0].color) {
                .color => |color| color,
                .texture => |texture| blk: {
                    const bs = tri[1];
                    var texcoord: @Vector(2, f32) = @splat(0);
                    for (0..3) |i| {
                        texcoord += @as(@Vector(2, f32), @splat(bs[i])) * tri[0].vs[i].texcoord;
                    }
                    break :blk std.simd.extract(texture.get(texcoord[0], texcoord[1]), 0, 3);
                }
            },
        };
        const material = switch (intersection.object) {
            .sphere => |sph| sph[0].material,
            .plane => |pl| pl.material,
            .triangle => |tri| tri[0].material,
        };
        if (material.roughness > 0) {
            for (0..3) |i| {
                normal[i] += generateGaussian() * material.roughness;
            }
        }
        normal = util.normalize(normal);
        
        if (gi < State.gi) {
            var ray2: Ray = .{.origin = p, .n = util.normalize(generateUnitSphere() + normal)};
            ray2.origin += scalarVec(secondary_ray_off, ray2.n);
            const color2 = ray2.trace(data, depth, gi + 1, inside);
            var lambert = @max(util.dot(normal, ray2.n), 0);
            inline for (0..3) |i| {
                pixel[i] += color2[i] * color[i] * lambert * material.diffuse[i];
            }
        }
        if (depth < State.bounces and util.greaterThan(material.shininess, 0)) {
            const reflected_ray = Ray {.origin = p, 
                    .n = util.normalize(ray.n - @as(Vec3, @splat(2*util.dot(normal, ray.n))) * normal) };
            const reflected_color = reflected_ray.trace(data, depth+1, gi, inside);
            inline for (0..3) |i| {
                pixel[i] += reflected_color[i] * material.shininess[i];
            }
            
        }
        // std.debug.print("{any}\n", .{material});
        if (util.greaterThan(material.diffuse, 0)) {
            for (data.lights.items) |light| {
                const light_dir = if (light.is_sun) light.dir else util.normalize(light.dir - p);
                const light_source_ray = Ray {.origin = p + scalarVec(secondary_ray_off, light_dir), .n = light_dir};
                if (light_source_ray.intersect(data)) |light_inter| {
                    if (light.is_sun) continue;
                    const mul = (light.dir - p)[0] / light_dir[0];
                    if (light_inter.t < mul) {
                        continue;
                    }
                    
                }
                var lambert = @max(util.dot(normal, light_dir), 0);
                if (!light.is_sun) lambert /= @reduce(.Add, (light.dir - p)*(light.dir - p));
                const new_color = color * light.color * @as(Vec3, @splat(lambert));
                inline for (0..3) |i| {
                    pixel[i] += new_color[i] * material.diffuse[i];
                }
            }
        }
        if (util.greaterThan(material.transparency, 0)) {
            // const index = switch (intersection.object) {
            //     .sphere => |sph| if (sph[1]) material.index else 1/material.index,
            //     else => 1/material.index,
            // };
            
            const index = if (inside) material.index else 1/material.index;
            // if (index != material.index) std.debug.print("inside sphere\n", .{});
            
            const ni = util.dot(normal, ray.n);
            const k = 1.0 - index*index * (1.0 - ni*ni);
            if (k < 0) {
                const reflected_ray = Ray {.origin = p, 
                    .n = util.normalize(ray.n - @as(Vec3, @splat(2*util.dot(normal, ray.n))) * normal) };
                const reflected_color = reflected_ray.trace(data, depth+1, gi, inside);
                inline for (0..3) |i| {
                    pixel[i] += reflected_color[i] * material.transparency[i];
                } 
            } else {
                const refracted_n = util.normalize(scalarVec(index, ray.n) - scalarVec(index*ni + @sqrt(k), normal));
                const refracted_ray = Ray {
                    .origin = p + scalarVec(secondary_ray_off, refracted_n), 
                    .n =  refracted_n,
                };
                // std.debug.print("ref {any} {any}\n", .{ray.n, refracted_ray.n});
                // std.debug.print("{any}{any}\n", .{ray.n, refracted_ray.n});
                const refracted_color = refracted_ray.trace(data, depth+1, gi,!inside);
                // std.debug.print("{any} {}\n", .{refracted_color, depth});
                inline for (0..3) |i| {
                    pixel[i] += refracted_color[i] * material.transparency[i];
                }
                
            }

        }
        return pixel;
    }
};
pub fn getRayOnSubPixel(x: f32, y: f32) ?Ray {
    const max: f32 = @floatFromInt(@max(State.width, State.height));
    const xx = 2*x - @as(f32, @floatFromInt(State.width));
    const yy = @as(f32,@floatFromInt(State.height)) - 2*y;
    const sx = xx/max;
    const sy = yy/max;
    if (State.fisheye) {
        if (sx*sx + sy*sy > 1) return null;
        const mul: Vec3 = @splat(@sqrt(1 - sx*sx - sy*sy));
        const n = mul * State.f + @as(Vec3, @splat(sx)) * State.r + @as(Vec3, @splat(sy)) * State.u;
        return .{.origin = State.e, .n = util.normalize(n)};
    }
    if (State.panorama) {
        @panic("unimplemented");
    }
    const n = State.f + @as(Vec3, @splat(sx)) * State.r + @as(Vec3, @splat(sy)) * State.u;
    return .{.origin = State.e, .n = util.normalize(n)};
}
pub fn getRayOnPixel(x: usize, y: usize) ?Ray {

    const max: f32 = @floatFromInt(@max(State.width, State.height));
    const xx = @as(i32,@intCast(2*x)) - @as(i32, @intCast(State.width));
    const yy = @as(i32,@intCast(State.height)) - @as(i32, @intCast(2*y));
    const sx = @as(f32, @floatFromInt(xx)) / max;
    const sy = @as(f32, @floatFromInt(yy)) / max;
    if (State.fisheye) {
        if (sx*sx + sy*sy > 1) return null;
        const mul: Vec3 = @splat(@sqrt(1 - sx*sx - sy*sy));
        const n = mul * State.f + @as(Vec3, @splat(sx)) * State.r + @as(Vec3, @splat(sy)) * State.u;
        return .{.origin = State.e, .n = util.normalize(n)};
    }
    if (State.panorama) {
        @panic("unimplemented");
    }
    const n = State.f + @as(Vec3, @splat(sx)) * State.r + @as(Vec3, @splat(sy)) * State.u;
    return .{.origin = State.e, .n = util.normalize(n)};
}
fn getShininess(obj: anytype) Vec3 {
    return obj.shininess;
}
pub fn generateGaussian() f32 {
    const v1 = RandGen.random().float(f32);
    const v2 = RandGen.random().float(f32);
    return @sqrt(-2.0*@log(v1)) * @cos(2.0*std.math.pi*v2);
}
pub fn generateUnitSphere() Vec3 {
    var v: Vec3 = undefined;
    for (0..3) |i| {
        v[i] = RandGen.random().float(f32);
    }
    while (@reduce(.Add, v*v) > 1) {
        for (0..3) |i| {
            v[i] = RandGen.random().float(f32);
        }
    }
    
    return v;
}
const Texture = struct {
    w: usize,
    h: usize,
    data: [*c]u32,
    pub fn get(self: Texture, u: f32, v: f32) Vec4 {
        const x: usize = @intFromFloat(u * @as(f32, @floatFromInt(self.w)));
        const y: usize = @intFromFloat(v * @as(f32, @floatFromInt(self.h)));
        // std.debug.print("{} {} {} {} {} {}\n", .{self.w,self.h,u,v,x,y});

        const color32 = util.Color4.fromU32(self.data[y * self.w + x]);
        var colorf4: Vec4 = undefined;
        const tpye_info = @typeInfo(util.Color4);
        inline for (tpye_info.Struct.fields, 0..) |f, i| {
            colorf4[i] = @as(f32, @floatFromInt(@field(color32, f.name))) / 255.0;
        }
        inline for (0..3) |i| {
            colorf4[i] = if (colorf4[i] <= 0.04045) colorf4[i]/12.92 else std.math.pow(f32, ((colorf4[i] + 0.055) / 1.055), 2.4); 
        }
        return colorf4;
    }
};
const Material = struct {
    shininess: Vec3 = .{0,0,0},
    diffuse: Vec3 = .{1,1,1},
    transparency: Vec3 = .{0,0,0},
    index: f32 = 1.458, // used for refraction
    roughness: f32 = 0,
    pub fn actual(self: Material) Material {
        var m = self;
        m.transparency = (@as(Vec3, @splat(1)) - self.shininess) * self.transparency;
        m.diffuse = @as(Vec3, @splat(1)) - m.shininess - m.transparency;
        return m;
    }
};

const ColorData = union(enum) {
    color: Vec3,
    texture: Texture,
};
const Sphere = struct {
    c: Vec3 = .{0,0,0},
    r: f32 = 0,
    color: ColorData,
    material: Material,
    aabb: AABB,
};
const Plane = struct {
    n: Vec3,
    p: Vec3,
    color: Vec3,
    material: Material,
};
const Triangle = struct {
    vs: [3]Vertex,
    e1: Vec3,
    e2: Vec3,
    n: Vec3,
    color: ColorData,
    material: Material,
    aabb: AABB,
};
const Geom = union(enum) {
    triangle: Triangle,
    sphere: Sphere,

    pub fn getAABB(self: Geom) AABB {
        return switch (self) {
            .sphere => |s| s.aabb,
            .triangle => |t| t.aabb,
        };
    }
};

const LightSource = struct {
    dir: Vec3,
    color: Vec3,
    is_sun: bool,
};
const Vertex = struct {
    v: Vec3,
    texcoord: @Vector(2, f32),
};
const SphereList = std.ArrayList(Sphere);
const LightList = std.ArrayList(LightSource);
const PlaneList = std.ArrayList(Plane);
const VertexList = std.ArrayList(Vertex);
const TriangleList = std.ArrayList(Triangle);
const GeomList = std.ArrayList(Geom);
const BVHList = std.ArrayList(BVHNode);
const Data = struct { 
    vertexes: VertexList,
    geoms: GeomList,
    planes: PlaneList,
    lights: LightList,
    bvh_tree: BVHList,
    root_index: u32,
    pub fn deinit(self: Data) void {
        const type_info = @typeInfo(Data);
        inline for (type_info.Struct.fields) |f| {
            if (@hasDecl(f.type, "deinit")) @field(self, f.name).deinit();
        }
    }
};
const AABB = struct {
    min: Vec3,
    max: Vec3,
    pub fn lessThan(self: AABB, other: AABB, axis: u32) bool {
        return self.max[axis] < other.min[axis];
    }
};
const BVHNode = struct {
    pub var node_index: u32 = 0;
    pub fn makeTree(data: *Data) u32 {
        return makeRecursive(data.geoms.items, 0, data.bvh_tree.items, 0);
    }
    pub fn isLeaf(self: BVHNode) bool {
        return self.geoms.len <= 5;
    }
    fn makeRecursive(geoms: []Geom, axis: u32, node_pool: []BVHNode, ct: u32) u32 {
        const root_index = node_index;
        const root = &node_pool[root_index];
        node_index += 1;
        root.geoms = geoms;
        root.aabb = BVHNode.getAABBOfMany(root.geoms);
        root.right = std.math.maxInt(u32);
        root.left = std.math.maxInt(u32);
        if (geoms.len > 5) {
            const mid_index: u32 = @intCast(partition(root.geoms, (root.aabb.max[axis] + root.aabb.min[axis])/2, axis) + 1);
            root.left = makeRecursive(geoms[0..mid_index], (axis+1)%3, node_pool, ct);
            if (mid_index == 0) {
                if (ct >= 3) return root_index;
                root.right = makeRecursive(geoms[mid_index..], (axis+1)%3, node_pool, ct+1);
            } else {
                root.right = makeRecursive(geoms[mid_index..], (axis+1)%3, node_pool, ct);
            }
        }
 
        return root_index;
    }
    // return the last index of the left partition
    pub fn partition(geoms: []Geom, mid: f32, axis: u32) i64 {
        var small_index: i64 = -1;
        for (geoms, 0..) |g, i| {
            if (g.getAABB().max[axis] < mid) {
                small_index += 1;
                std.mem.swap(Geom, &geoms[i], &geoms[@intCast(small_index)]);
            } 
        }
        return small_index;

    }
    pub fn getAABBOfMany(geoms: []Geom) AABB {
        var aabb = AABB {.max = @splat(-std.math.floatMax(f32)), .min = @splat(std.math.floatMax(f32)) };
        for (geoms) |g| {
            // std.debug.print("g aabb: {any} {any}\n", .{g.getAABB().max, aabb.max});
            aabb.max = @max(g.getAABB().max, aabb.max);
            aabb.min = @min(g.getAABB().min, aabb.min);
        }
        return aabb;
    }
   
    geoms: []Geom,
    left: usize,
    right: usize,
    aabb: AABB,

};
fn startsWith(s: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, s, prefix);
}
fn splitSpace(s: []const u8) std.mem.TokenIterator(u8, .scalar) {
    return std.mem.tokenizeScalar(u8, s, ' ');
}



pub fn main() !void {
    RandGen.seed( @intCast(std.time.timestamp()));
    const allocator = std.heap.c_allocator;
    // init data structures
    // parse args
    const args = try std.process.argsAlloc(allocator);
    // defer std.process.argsFree(allocator, args);

    if (args.len <= 1) {
        std.log.err("No Input File Provided\n", .{});
        return;
    }
    // parse input file
    var data: Data = .{
            .planes = PlaneList.init(allocator),
            .lights = LightList.init(allocator), 
            .vertexes = VertexList.init(allocator),
            .geoms = GeomList.init(allocator),
            .bvh_tree = BVHList.init(allocator),
            .root_index = 0,
    };
    // defer data.deinit();
    
    try parseCommands(args[1], allocator, &data);
    defer {
        if (State.output_path.len != 0) allocator.free(State.output_path);
    }
    try data.bvh_tree.resize(data.geoms.items.len);
    std.debug.print("Geom[{}], Planes[{}]\n", .{data.geoms.items.len, data.planes.items.len});
    data.root_index = BVHNode.makeTree(&data);
    std.debug.print("BVH tree built with {} nodes\n", .{BVHNode.node_index});
    std.debug.print("Image[{} * {}] => {s}\n", .{State.width, State.height, State.output_path});
    var cv_buf = try allocator.alloc(u32, State.width * State.height);
    var rgb_buf = try allocator.alloc(Vec4, State.width * State.height);
    @memset(cv_buf, 0);
    @memset(rgb_buf, .{0,0,0,0});
    // defer allocator.free(cv_buf);
    // defer allocator.free(rgb_buf);
    


    for (0..State.height) |y| {
        for (0..State.width) |x| {
            const pixel = &rgb_buf[y * State.width + x];
            if (State.aa > 0) {
                for (0..State.aa) |_| {
                    const xa = @as(f32, @floatFromInt(x)) + RandGen.random().float(f32);
                    const ya = @as(f32, @floatFromInt(y)) + RandGen.random().float(f32);
                    var ray = getRayOnSubPixel(xa, ya) orelse continue;
                    if (State.dof[1] != 0) {
                        ray = ray.dof();
                    }
                    const color = ray.trace(data, 0, 0, false);
                    pixel.* += color;
                }
                pixel.* /= @splat(@floatFromInt(State.aa));

            } else {
                var ray = getRayOnPixel(x, y) orelse continue;
                if (State.dof[1] != 0) {
                    ray = ray.dof();
                }
                const color = ray.trace(data, 0, 0, false);
                pixel.* += color;
            }


        }
    }
    // const ray = getRayOnPixel(60, 25);
    // std.debug.print("ray: {any} == {any}", .{ray, util.normalize(Vec3{0.2,0,-1})});
    // if (ray.intersect(data)) |_| {
    //     std.debug.print("intersect\n", .{});
    // }
    // converting lrgb -> exposed -> srgb
    for (0..State.height) |y| {
        for (0..State.width) |x| {
            const lcolor = @min(@max(rgb_buf[y * State.width + x], @as(Vec4, @splat(0))), @as(Vec4, @splat(1)));
            var scolor = lcolor;

            for (0..3) |i| {
                if (State.expose) |expose| {
                    scolor[i] = 1.0 - @exp(-expose * scolor[i]);
                }
                if (scolor[i] <= 0.0031308) {
                    scolor[i] *= 12.92;
                } else {
                    scolor[i] = 1.055 * std.math.pow(f32, scolor[i], 1.0/2.4) - 0.055;
                }
            }
            cv_buf[y * State.width + x] = util.rgbaVecToU32(scolor);
            
        }
    }
    const res = lodepng.lodepng_encode32_file(State.output_path, @ptrCast(cv_buf.ptr), State.width, State.height);
    std.debug.print("PNG encode: [{}] {s}\n", .{res, lodepng.lodepng_error_text(res)});
}