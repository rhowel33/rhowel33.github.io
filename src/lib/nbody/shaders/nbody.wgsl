// Direct-summation N-body force kernel + leapfrog kick/drift, plus the
// Barnes-Hut tree-walk force kernel. Mirrors shaders/nbody.metal.
//
// Bindings:
//   0: positions      (xyz, mass)   — storage, read_write
//   1: velocities     (vxvyvz, _)   — storage, read_write
//   2: accelerations  (axayaz, _)   — storage, read_write
//   3: SimConstants uniform
//   4: Barnes-Hut tree nodes        — storage, read  (unused in direct mode)
//
// Workgroup size is 256 (required for the 256-wide tile in forces_direct).

const WG: u32 = 256u;

struct Halo {
  center_v0sq: vec4<f32>,   // xyz = center, w = v0²
  rc_sq:       vec4<f32>,   // x   = rc²
};

struct SimConstants {
  softening_sq: f32,
  G:            f32,
  dt:           f32,
  half_dt:      f32,
  c:            f32,
  pn_amp:       f32,
  theta_sq:     f32,
  _padf:        f32,
  n_stars:      u32,
  n_sinks:      u32,
  n_bodies:     u32,
  n_halos:      u32,
  halos:        array<Halo, 5>,
  // Halo stride is 32, so array<Halo, 5> = 160 bytes starting at offset 48 and
  // ending at 208. Trailing u32 + 16-byte struct alignment bumps total size to
  // 224 bytes. Keep the host-side UNI_BYTES/uniU32 indices in sync.
  n_bodies_padded: u32,
};

// One node of the LBVH. Leaves live at [n_bodies-1 .. 2·n_bodies-2],
// internal at [0 .. n_bodies-2]. The `left` field doubles as body index on
// leaves (so force walk doesn't need to round-trip through morton_idx).
struct LBVHNode {
  com_mass: vec4<f32>,    // xyz = COM, w = mass
  aabb_min: vec4<f32>,    // xyz = AABB min (w unused)
  aabb_max: vec4<f32>,    // xyz = AABB max (w unused)
  left:     i32,          // internal: left child index; leaf: body index
  right:    i32,          // internal: right child index; leaf: unused
  parent:   i32,          // -1 for root
  visited:  atomic<u32>,  // bottom-up summarize counter
};

@group(0) @binding(0) var<storage, read_write> positions:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             k:             SimConstants;

// GPU-LBVH scratch. Each slot represents a monotonically-encoded float so
// atomicMin/atomicMax over u32 gives the correct float ordering.
//   bbox[0..2] = min xyz   (init to 0xFFFFFFFF)
//   bbox[3..5] = max xyz   (init to 0x00000000)
@group(0) @binding(5) var<storage, read_write> bbox:          array<atomic<u32>, 6>;
@group(0) @binding(6) var<storage, read_write> morton_keys:   array<u32>;
@group(0) @binding(7) var<storage, read_write> morton_idx:    array<u32>;
@group(0) @binding(8) var<storage, read_write> lbvh_nodes:    array<LBVHNode>;

// --- Monotonic f32 ↔ u32 ordering, for atomic min/max on floats ---------

fn f32_to_u32_ord(f: f32) -> u32 {
  let u = bitcast<u32>(f);
  // Negatives: flip all bits → more-negative sorts smaller.
  // Non-negatives: flip sign bit → all larger than any encoded negative.
  if ((u & 0x80000000u) != 0u) { return ~u; }
  return u ^ 0x80000000u;
}

fn u32_ord_to_f32(u: u32) -> f32 {
  if ((u & 0x80000000u) == 0u) { return bitcast<f32>(~u); }
  return bitcast<f32>(u ^ 0x80000000u);
}

var<workgroup> tile: array<vec4<f32>, 256>;

fn halo_contribution(my_pos: vec3<f32>, acc_in: vec3<f32>) -> vec3<f32> {
  var acc = acc_in;
  for (var h: u32 = 0u; h < k.n_halos; h = h + 1u) {
    let halo = k.halos[h];
    let rvec = my_pos - halo.center_v0sq.xyz;
    let r2h  = dot(rvec, rvec) + halo.rc_sq.x;
    acc = acc - (halo.center_v0sq.w / r2h) * rvec;
  }
  return acc;
}

@compute @workgroup_size(256)
fn forces_direct(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
) {
  let g = gid.x;
  let l = lid.x;

  var my_pos = vec4<f32>(0.0);
  if (g < k.n_bodies) { my_pos = positions[g]; }

  var acc = vec3<f32>(0.0);

  var tile_start: u32 = 0u;
  loop {
    if (tile_start >= k.n_bodies) { break; }

    let load_idx = tile_start + l;
    if (load_idx < k.n_bodies) {
      tile[l] = positions[load_idx];
    } else {
      tile[l] = vec4<f32>(0.0);
    }
    workgroupBarrier();

    let remain = k.n_bodies - tile_start;
    var tile_end = WG;
    if (remain < WG) { tile_end = remain; }
    for (var j: u32 = 0u; j < tile_end; j = j + 1u) {
      let r_vec  = tile[j].xyz - my_pos.xyz;
      let m      = tile[j].w;
      let r2     = dot(r_vec, r_vec) + k.softening_sq;
      let inv_r  = inverseSqrt(r2);
      let inv_r3 = inv_r * inv_r * inv_r;
      acc = acc + (k.G * m * inv_r3) * r_vec;
    }
    workgroupBarrier();

    tile_start = tile_start + WG;
  }

  if (g < k.n_bodies) {
    acc = halo_contribution(my_pos.xyz, acc);
    accelerations[g] = vec4<f32>(acc, 0.0);
  }
}

// Iterative DFS walk of the GPU-built LBVH with the Barnes-Hut opening
// criterion (s² ≥ θ² r² ⇒ descend), where s is the longest AABB extent.
@compute @workgroup_size(256)
fn forces_lbvh(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies) { return; }

  let my_pos = positions[g];
  var acc = vec3<f32>(0.0);

  // Binary radix tree: depth ≤ 2·log₂(N). Stack of 64 fits N ≤ 2^31.
  var stack: array<i32, 64>;
  var sp: i32 = 0;
  stack[sp] = 0;
  sp = sp + 1;

  let leaf_base = i32(k.n_bodies) - 1;

  loop {
    if (sp <= 0) { break; }
    sp = sp - 1;
    let idx = stack[sp];

    // Read fields individually — LBVHNode contains an atomic<u32>, and WGSL
    // forbids binding a struct that contains an atomic to a `let`/`var`.
    let cm = lbvh_nodes[idx].com_mass;
    let m  = cm.w;
    if (m > 0.0) {
      let com = cm.xyz;
      let d   = com - my_pos.xyz;
      let r2  = dot(d, d);

      let is_leaf = idx >= leaf_base;
      let mn = lbvh_nodes[idx].aabb_min.xyz;
      let mx = lbvh_nodes[idx].aabb_max.xyz;
      let extent = mx - mn;
      let s  = max(extent.x, max(extent.y, extent.z));
      let s2 = s * s;

      let open = (!is_leaf) && (s2 >= k.theta_sq * r2);
      if (open) {
        let l_child = lbvh_nodes[idx].left;
        let r_child = lbvh_nodes[idx].right;
        if (sp < 63) { stack[sp] = l_child; sp = sp + 1; }
        if (sp < 63) { stack[sp] = r_child; sp = sp + 1; }
      } else {
        var is_self = false;
        if (is_leaf) { is_self = lbvh_nodes[idx].left == i32(g); }
        if (!is_self) {
          let r2_soft = r2 + k.softening_sq;
          let inv_r   = inverseSqrt(r2_soft);
          let inv_r3  = inv_r * inv_r * inv_r;
          acc = acc + (k.G * m * inv_r3) * d;
        }
      }
    }
  }

  acc = halo_contribution(my_pos.xyz, acc);
  accelerations[g] = vec4<f32>(acc, 0.0);
}

// Post-Newtonian sink–sink correction. Adds 1PN + 2.5PN contributions to
// each sink's acceleration via the relative-orbit expressions, then
// partitions mass-weightedly between the two bodies. See the long comment
// in shaders/nbody.metal for the form of A_1PN, B_1PN, A_2.5, B_2.5.
@compute @workgroup_size(64)
fn pn_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= k.n_sinks) { return; }
  let i = k.n_stars + gid.x;

  let pi = positions[i];
  let vi = velocities[i];
  let mi = pi.w;
  if (mi <= 0.0) { return; }

  var a_corr = vec3<f32>(0.0);
  let c2_inv = 1.0 / (k.c * k.c);
  let c5_inv = c2_inv * c2_inv / k.c;

  for (var s: u32 = 0u; s < k.n_sinks; s = s + 1u) {
    if (s == gid.x) { continue; }
    let j  = k.n_stars + s;
    let pj = positions[j];
    let vj = velocities[j];
    let mj = pj.w;
    if (mj <= 0.0) { continue; }

    let Delta = pi.xyz - pj.xyz;
    let r     = length(Delta);
    if (r < 1e-6) { continue; }
    let nhat  = Delta / r;
    let vrel  = vi.xyz - vj.xyz;
    let v2    = dot(vrel, vrel);
    let rdot  = dot(nhat, vrel);

    let M       = mi + mj;
    let eta     = (mi * mj) / (M * M);
    let GM      = k.G * M;
    let GMoverR = GM / r;

    let A1 = 2.0 * (2.0 + eta) * GMoverR
           - (1.0 + 3.0 * eta) * v2
           + 1.5 * eta * rdot * rdot;
    let B1 = -2.0 * (2.0 - eta) * rdot;

    let A25 = -(8.0 / 5.0) * eta * GMoverR * rdot
              * (3.0 * v2 + (17.0 / 3.0) * GMoverR);
    let B25 =  (8.0 / 5.0) * eta * GMoverR
              * (v2 + 3.0 * GMoverR);

    let coeff_n = A1 * c2_inv + k.pn_amp * A25 * c5_inv;
    let coeff_v = B1 * c2_inv + k.pn_amp * B25 * c5_inv;

    let a_rel = -(GM / (r * r)) * (nhat * coeff_n + vrel * coeff_v);
    a_corr = a_corr + (mj / M) * a_rel;
  }

  accelerations[i] = vec4<f32>(accelerations[i].xyz + a_corr, 0.0);
}

@compute @workgroup_size(256)
fn kick(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies) { return; }
  var v = velocities[g];
  let a = accelerations[g].xyz;
  v = vec4<f32>(v.xyz + a * k.half_dt, v.w);
  velocities[g] = v;
}

@compute @workgroup_size(256)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies) { return; }
  var p = positions[g];
  let v = velocities[g].xyz;
  p = vec4<f32>(p.xyz + v * k.dt, p.w);
  positions[g] = p;
}

// ─── GPU-LBVH: bounding box ─────────────────────────────────────────────
//
// Single pass. Each thread reduces its live body into the bbox via atomic
// min/max on the monotonic-encoded u32 representation. Dead particles
// (mass ≤ 0) are skipped so they don't expand the box.

@compute @workgroup_size(256)
fn bbox_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 6u) { return; }
  // Indices 0..2 = min (seed with +inf encoded), 3..5 = max (seed with -inf encoded).
  if (gid.x < 3u) {
    atomicStore(&bbox[gid.x], 0xFFFFFFFFu);
  } else {
    atomicStore(&bbox[gid.x], 0x00000000u);
  }
}

@compute @workgroup_size(256)
fn bbox_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies) { return; }
  let p = positions[g];
  if (p.w <= 0.0) { return; }
  let ex = f32_to_u32_ord(p.x);
  let ey = f32_to_u32_ord(p.y);
  let ez = f32_to_u32_ord(p.z);
  atomicMin(&bbox[0u], ex);
  atomicMin(&bbox[1u], ey);
  atomicMin(&bbox[2u], ez);
  atomicMax(&bbox[3u], ex);
  atomicMax(&bbox[4u], ey);
  atomicMax(&bbox[5u], ez);
}

// ─── GPU-LBVH: Morton codes ─────────────────────────────────────────────
//
// 30-bit Morton (10 bits per axis), packed in a u32 so sorting is cheap.
// Positions are normalized to [0, 1024) within the bbox. Dead particles
// get a sentinel key (0xFFFFFFFF) so they sort to the end of the array
// and never participate in the tree.

fn spread_bits_10(v: u32) -> u32 {
  // Expand 10-bit value x so each bit has two zeros between it.
  var x = v & 0x000003FFu;
  x = (x | (x << 16u)) & 0x030000FFu;
  x = (x | (x <<  8u)) & 0x0300F00Fu;
  x = (x | (x <<  4u)) & 0x030C30C3u;
  x = (x | (x <<  2u)) & 0x09249249u;
  return x;
}

fn morton3d(xn: f32, yn: f32, zn: f32) -> u32 {
  let xq = u32(clamp(xn * 1024.0, 0.0, 1023.0));
  let yq = u32(clamp(yn * 1024.0, 0.0, 1023.0));
  let zq = u32(clamp(zn * 1024.0, 0.0, 1023.0));
  return spread_bits_10(xq) | (spread_bits_10(yq) << 1u) | (spread_bits_10(zq) << 2u);
}

@compute @workgroup_size(256)
fn compute_morton(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies_padded) { return; }

  // Padding slots (beyond n_bodies): sentinel keys so the sort puts them at
  // the end of the array and they don't end up in the tree.
  if (g >= k.n_bodies) {
    morton_keys[g] = 0xFFFFFFFFu;
    morton_idx[g]  = 0xFFFFFFFFu;
    return;
  }

  let p = positions[g];
  morton_idx[g] = g;

  if (p.w <= 0.0) {
    morton_keys[g] = 0xFFFFFFFFu;
    return;
  }

  let mn = vec3<f32>(
    u32_ord_to_f32(atomicLoad(&bbox[0u])),
    u32_ord_to_f32(atomicLoad(&bbox[1u])),
    u32_ord_to_f32(atomicLoad(&bbox[2u])),
  );
  let mx = vec3<f32>(
    u32_ord_to_f32(atomicLoad(&bbox[3u])),
    u32_ord_to_f32(atomicLoad(&bbox[4u])),
    u32_ord_to_f32(atomicLoad(&bbox[5u])),
  );
  let size = max(mx - mn, vec3<f32>(1e-6, 1e-6, 1e-6));
  let nrm  = (p.xyz - mn) / size;

  morton_keys[g] = morton3d(nrm.x, nrm.y, nrm.z);
}

// ─── GPU-LBVH: bitonic sort ─────────────────────────────────────────────
//
// Sorts (morton_keys, morton_idx) pairs in ascending key order. One
// dispatch per bitonic stage (k_stage, j_step); stages are driven from
// the host via a dynamic uniform offset. All log²(N_padded) stages run
// inside one compute pass — inter-dispatch storage hazards make WebGPU
// insert the necessary barriers automatically.

struct SortParams {
  k_stage: u32,
  j_step:  u32,
};

@group(1) @binding(0) var<uniform> sort_params: SortParams;

@compute @workgroup_size(256)
fn bitonic_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= k.n_bodies_padded) { return; }

  let l = i ^ sort_params.j_step;
  if (l <= i) { return; }

  let asc = (i & sort_params.k_stage) == 0u;
  let ki = morton_keys[i];
  let kl = morton_keys[l];

  var swap = false;
  if (asc) { swap = ki > kl; } else { swap = ki < kl; }

  if (swap) {
    morton_keys[i] = kl;
    morton_keys[l] = ki;
    let ii = morton_idx[i];
    morton_idx[i] = morton_idx[l];
    morton_idx[l] = ii;
  }
}

// ─── GPU-LBVH: Karras hierarchy + bottom-up summarize ───────────────────

// Common-prefix length of morton_keys[a] and morton_keys[b], with an index
// tiebreaker so identical keys still give a total order.
fn bvh_delta(a: i32, b: i32, n: i32) -> i32 {
  if (b < 0 || b >= n) { return -1; }
  let ma = morton_keys[a];
  let mb = morton_keys[b];
  if (ma == mb) {
    return i32(countLeadingZeros(u32(a) ^ u32(b))) + 32;
  }
  return i32(countLeadingZeros(ma ^ mb));
}

// Zero atomic counters and seed root.parent = -1 before each build.
@compute @workgroup_size(256)
fn lbvh_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  let total = select(2u * k.n_bodies - 1u, 0u, k.n_bodies == 0u);
  if (g >= total) { return; }
  atomicStore(&lbvh_nodes[g].visited, 0u);
  if (g == 0u) {
    lbvh_nodes[0].parent = -1;
  }
}

// Tero Karras, "Maximizing Parallelism in the Construction of BVHs,
// Octrees, and k-d Trees", HPG 2012. One thread per internal node.
// Produces an implicit binary radix tree over the sorted Morton keys.
@compute @workgroup_size(256)
fn build_hierarchy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let N = i32(k.n_bodies);
  if (N <= 1 || i >= N - 1) { return; }

  // Determine direction of range
  var d: i32 = 1;
  if (i > 0) {
    let da = bvh_delta(i, i + 1, N);
    let db = bvh_delta(i, i - 1, N);
    d = select(-1, 1, da >= db);
  }

  // Upper bound on range length
  let delta_min = bvh_delta(i, i - d, N);
  var l_max: i32 = 2;
  loop {
    if (bvh_delta(i, i + l_max * d, N) <= delta_min) { break; }
    l_max = l_max * 2;
  }

  // Binary search for the exact range length
  var l: i32 = 0;
  var t: i32 = l_max / 2;
  loop {
    if (t <= 0) { break; }
    if (bvh_delta(i, i + (l + t) * d, N) > delta_min) { l = l + t; }
    t = t / 2;
  }
  let j = i + l * d;

  // Binary search for the split position within [i, j]
  let delta_node = bvh_delta(i, j, N);
  var s: i32 = 0;
  var div: i32 = 2;
  loop {
    let tt = (l + div - 1) / div;     // ceil(l / div)
    if (bvh_delta(i, i + (s + tt) * d, N) > delta_node) { s = s + tt; }
    if (tt <= 1) { break; }
    div = div * 2;
  }
  let gamma = i + s * d + min(d, 0);

  let lo = min(i, j);
  let hi = max(i, j);
  var left_child: i32;
  var right_child: i32;
  if (lo == gamma)      { left_child  = gamma + (N - 1); }       // leaf
  else                  { left_child  = gamma;          }        // internal
  if (hi == gamma + 1)  { right_child = gamma + 1 + (N - 1); }   // leaf
  else                  { right_child = gamma + 1;      }        // internal

  lbvh_nodes[i].left          = left_child;
  lbvh_nodes[i].right         = right_child;
  lbvh_nodes[left_child].parent  = i;
  lbvh_nodes[right_child].parent = i;
}

// Multi-pass bottom-up summarize. Split into two kernels so inter-dispatch
// storage barriers — which WebGPU inserts automatically between compute
// dispatches — do the cross-workgroup synchronization that a single-kernel
// sibling-climb can't portably rely on.
//
//   1. summarize_leaves         — one pass, populates every leaf.
//   2. summarize_internal_pass  — one thread per internal node; aggregates
//                                 iff both children are ready (visited != 0)
//                                 and sets visited. Dispatched K times with
//                                 K ≥ tree depth so the wave reaches root.
//
// `visited` doubles as the "is this node summarized yet" flag.

@compute @workgroup_size(256)
fn summarize_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= k.n_bodies) { return; }

  let leaf_idx = i32(g) + i32(k.n_bodies) - 1;
  let body_idx = morton_idx[g];

  if (body_idx >= k.n_bodies) {
    lbvh_nodes[leaf_idx].com_mass = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    lbvh_nodes[leaf_idx].aabb_min = vec4<f32>( 1e30,  1e30,  1e30, 0.0);
    lbvh_nodes[leaf_idx].aabb_max = vec4<f32>(-1e30, -1e30, -1e30, 0.0);
    lbvh_nodes[leaf_idx].left     = -1;
  } else {
    let p = positions[body_idx];
    lbvh_nodes[leaf_idx].com_mass = vec4<f32>(p.xyz, p.w);
    lbvh_nodes[leaf_idx].aabb_min = vec4<f32>(p.xyz, 0.0);
    lbvh_nodes[leaf_idx].aabb_max = vec4<f32>(p.xyz, 0.0);
    lbvh_nodes[leaf_idx].left     = i32(body_idx);
  }
  atomicStore(&lbvh_nodes[leaf_idx].visited, 1u);
}

@compute @workgroup_size(256)
fn summarize_internal_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  // N internal nodes are indexed [0 .. N-2].
  if (g + 1u >= k.n_bodies) { return; }
  let idx = i32(g);

  if (atomicLoad(&lbvh_nodes[idx].visited) != 0u) { return; }

  let li = lbvh_nodes[idx].left;
  let ri = lbvh_nodes[idx].right;
  if (atomicLoad(&lbvh_nodes[li].visited) == 0u) { return; }
  if (atomicLoad(&lbvh_nodes[ri].visited) == 0u) { return; }

  let L_cm = lbvh_nodes[li].com_mass;
  let R_cm = lbvh_nodes[ri].com_mass;
  let L_mn = lbvh_nodes[li].aabb_min;
  let R_mn = lbvh_nodes[ri].aabb_min;
  let L_mx = lbvh_nodes[li].aabb_max;
  let R_mx = lbvh_nodes[ri].aabb_max;

  let mS = L_cm.w + R_cm.w;
  var com: vec3<f32> = vec3<f32>(0.0);
  if (mS > 0.0) { com = (L_cm.xyz * L_cm.w + R_cm.xyz * R_cm.w) / mS; }
  lbvh_nodes[idx].com_mass = vec4<f32>(com, mS);
  lbvh_nodes[idx].aabb_min = vec4<f32>(min(L_mn.xyz, R_mn.xyz), 0.0);
  lbvh_nodes[idx].aabb_max = vec4<f32>(max(L_mx.xyz, R_mx.xyz), 0.0);
  atomicStore(&lbvh_nodes[idx].visited, 1u);
}
