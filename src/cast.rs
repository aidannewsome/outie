//! The rays, cast by Embree, Intel's ray tracer, the one the paper's code uses: a mesh handed over once, then any number
//! of rays cast against it from many threads at once. Only the handful of Embree's functions this needs are declared,
//! by hand, against Embree 4's C interface. A ray never stops at the triangle it leaves, as the paper's code drops a hit
//! on its own face: each ray carries its triangle's number, and a filter turns that triangle's hits away.

use std::ffi::{c_char, c_void};

const GEOMETRY_TRIANGLE: u32 = 0;
const BUFFER_INDEX: u32 = 0;
const BUFFER_VERTEX: u32 = 1;
const FORMAT_UINT3: u32 = 0x5003;
const FORMAT_FLOAT3: u32 = 0x9003;
const NO_HIT: u32 = u32::MAX;

/// Embree's RTCRayHit: a ray, and where it first hit.
#[repr(C, align(16))]
struct RayHit {
    origin: [f32; 3],
    near: f32,
    direction: [f32; 3],
    time: f32,
    far: f32,
    mask: u32,
    id: u32, // the triangle the ray leaves
    flags: u32,
    normal: [f32; 3],
    u: f32,
    v: f32,
    triangle: u32,
    geometry: u32,
    instance: u32,
    instance_triangle: u32,
}

/// Embree's RTCFilterFunctionNArguments, for one ray at a time.
#[repr(C)]
struct FilterArguments {
    valid: *mut i32,
    geometry_user: *mut c_void,
    context: *mut c_void,
    ray: *const RayHit, // for one ray, a ray and its hit are laid out as RTCRayHit is
    hit: *const [u32; 6],
    n: u32,
}

unsafe extern "C" {
    fn rtcNewDevice(config: *const c_char) -> *mut c_void;
    fn rtcReleaseDevice(device: *mut c_void);
    fn rtcNewScene(device: *mut c_void) -> *mut c_void;
    fn rtcReleaseScene(scene: *mut c_void);
    fn rtcNewGeometry(device: *mut c_void, kind: u32) -> *mut c_void;
    fn rtcSetNewGeometryBuffer(
        geometry: *mut c_void,
        kind: u32,
        slot: u32,
        format: u32,
        stride: usize,
        count: usize,
    ) -> *mut c_void;
    fn rtcSetGeometryIntersectFilterFunction(
        geometry: *mut c_void,
        filter: unsafe extern "C" fn(*const FilterArguments),
    );
    fn rtcCommitGeometry(geometry: *mut c_void);
    fn rtcAttachGeometry(scene: *mut c_void, geometry: *mut c_void) -> u32;
    fn rtcReleaseGeometry(geometry: *mut c_void);
    fn rtcCommitScene(scene: *mut c_void);
    fn rtcIntersect1(scene: *mut c_void, rayhit: *mut RayHit, args: *mut c_void);
}

/// Turns away a ray's hit on the triangle it leaves.
unsafe extern "C" fn skip_own(args: *const FilterArguments) {
    unsafe {
        let args = &*args;
        if (*args.hit)[5] == (*args.ray).id {
            *args.valid = 0;
        }
    }
}

/// A mesh handed to Embree, ready for rays.
pub struct Scene {
    device: *mut c_void,
    scene: *mut c_void,
}

// Embree's scenes may be traced from many threads at once once committed.
unsafe impl Send for Scene {}
unsafe impl Sync for Scene {}

impl Scene {
    pub fn new(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Self {
        unsafe {
            let device = rtcNewDevice(std::ptr::null());
            assert!(!device.is_null(), "Embree could not start");
            let scene = rtcNewScene(device);
            let geometry = rtcNewGeometry(device, GEOMETRY_TRIANGLE);
            let corners = rtcSetNewGeometryBuffer(
                geometry,
                BUFFER_VERTEX,
                0,
                FORMAT_FLOAT3,
                12,
                vertices.len(),
            ) as *mut [f32; 3];
            for (i, v) in vertices.iter().enumerate() {
                *corners.add(i) = v.map(|x| x as f32);
            }
            let indices = rtcSetNewGeometryBuffer(
                geometry,
                BUFFER_INDEX,
                0,
                FORMAT_UINT3,
                12,
                triangles.len(),
            ) as *mut [u32; 3];
            for (i, t) in triangles.iter().enumerate() {
                *indices.add(i) = t.map(|c| c as u32);
            }
            rtcSetGeometryIntersectFilterFunction(geometry, skip_own);
            rtcCommitGeometry(geometry);
            rtcAttachGeometry(scene, geometry);
            rtcReleaseGeometry(geometry);
            rtcCommitScene(scene);
            Scene { device, scene }
        }
    }

    /// The first hit of a ray leaving a triangle, past any start: how far it travels, or None when it escapes.
    pub fn cast_ray(
        &self,
        own: usize,
        origin: [f64; 3],
        direction: [f64; 3],
        start: f32,
    ) -> Option<f32> {
        let mut ray = RayHit {
            origin: origin.map(|x| x as f32),
            near: start,
            direction: direction.map(|x| x as f32),
            time: 0.0,
            far: f32::INFINITY,
            mask: u32::MAX,
            id: own as u32,
            flags: 0,
            normal: [0.0; 3],
            u: 0.0,
            v: 0.0,
            triangle: NO_HIT,
            geometry: NO_HIT,
            instance: NO_HIT,
            instance_triangle: NO_HIT,
        };
        unsafe { rtcIntersect1(self.scene, &mut ray, std::ptr::null_mut()) };
        (ray.geometry != NO_HIT).then_some(ray.far)
    }
}

impl Drop for Scene {
    fn drop(&mut self) {
        unsafe {
            rtcReleaseScene(self.scene);
            rtcReleaseDevice(self.device);
        }
    }
}
