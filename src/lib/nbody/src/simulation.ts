import kernelSrc from '../shaders/nbody.wgsl?raw';
import type { HaloSpec } from './initial_conditions.ts';

export interface SimParams {
  dt:            number;
  softening:     number;
  G:             number;
  c:             number;
  pnAmp:         number;
  theta:         number;
  useBarnesHut:  boolean;
  captureRadius: number;
  halos?:        HaloSpec[];
}

export interface SinkInit {
  x:  number; y:  number; z:  number;
  vx: number; vy: number; vz: number;
  mass: number;
}

export interface SinkState {
  x:  number; y:  number; z:  number;
  vx: number; vy: number; vz: number;
  mass: number;
}

const WG_SIZE   = 256;
const MAX_HALOS = 5;
// 8 f32 + 4 u32 + 5·(2·vec4) + u32, padded to 16-byte struct alignment.
const UNI_BYTES = 224;
// WebGPU default minUniformBufferOffsetAlignment; used for packing bitonic
// stage params so setBindGroup can re-aim per dispatch.
const DYN_OFFSET_ALIGN = 256;

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return Math.max(p, 2);
}

export class Simulation {
  readonly nStars:         number;
  readonly nSinks:         number;
  readonly nBodies:        number;
  readonly nBodiesPadded:  number;

  readonly posBuf: GPUBuffer;
  readonly velBuf: GPUBuffer;
  readonly accBuf: GPUBuffer;

  // Latest sink state, updated after each stepMany for the renderer.
  sinks: SinkState[] = [];

  private readonly device:  GPUDevice;
  private readonly params:  SimParams;
  private readonly uniBuf:  GPUBuffer;
  private readonly uniData: ArrayBuffer;
  private readonly uniF32:  Float32Array;
  private readonly uniU32:  Uint32Array;

  private readonly forceDirectPipeline: GPUComputePipeline;
  private readonly forceLBVHPipeline:   GPUComputePipeline;
  private readonly pnPipeline:          GPUComputePipeline;
  private readonly kickPipeline:        GPUComputePipeline;
  private readonly driftPipeline:       GPUComputePipeline;
  private readonly bboxResetPipeline:   GPUComputePipeline;
  private readonly bboxReducePipeline:  GPUComputePipeline;
  private readonly mortonPipeline:      GPUComputePipeline;
  private readonly sortPipeline:        GPUComputePipeline;
  private readonly lbvhResetPipeline:   GPUComputePipeline;
  private readonly buildHierarchyPipeline:       GPUComputePipeline;
  private readonly summarizeLeavesPipeline:      GPUComputePipeline;
  private readonly summarizeInternalPassPipeline: GPUComputePipeline;

  private readonly bgLayout:        GPUBindGroupLayout;
  private readonly sortParamsLayout: GPUBindGroupLayout;
  private bindGroup!: GPUBindGroup;

  // Bitonic-sort parameter buffer (one uniform slot per stage, packed at
  // DYN_OFFSET_ALIGN so setBindGroup can re-aim per dispatch).
  private readonly sortParamsBuf:       GPUBuffer;
  private readonly sortParamsBindGroup: GPUBindGroup;
  private readonly sortStages:          ReadonlyArray<readonly [number, number]>;

  // GPU-LBVH scratch (phase 6a plumbing — kernels run but results are not
  // consumed yet; CPU Barnes-Hut still drives the force walk).
  private readonly bboxBuf:       GPUBuffer;   // 6 × u32 (min xyz, max xyz)
  private readonly mortonKeysBuf: GPUBuffer;   // nBodiesPadded × u32
  private readonly mortonIdxBuf:  GPUBuffer;   // nBodiesPadded × u32
  private readonly lbvhNodesBuf:  GPUBuffer;   // (2·nBodies − 1) × 64 bytes

  // CPU-side scratch for sink-only accretion + sink mirror (small readback).
  private readonly stagingPosBuf: GPUBuffer;
  private readonly stagingVelBuf: GPUBuffer;
  private readonly hostPos: Float32Array<ArrayBuffer>;
  private readonly hostVel: Float32Array<ArrayBuffer>;

  private accelPrimed = false;

  constructor(device: GPUDevice, nStars: number, nSinks: number, params: SimParams) {
    this.device        = device;
    this.nStars        = nStars;
    this.nSinks        = nSinks;
    this.nBodies       = nStars + nSinks;
    this.nBodiesPadded = nextPow2(this.nBodies);
    this.params        = { ...params };

    const bytes = this.nBodies * 4 * Float32Array.BYTES_PER_ELEMENT;
    const use   = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.posBuf = device.createBuffer({ size: bytes, usage: use });
    this.velBuf = device.createBuffer({ size: bytes, usage: use });
    this.accBuf = device.createBuffer({ size: bytes, usage: use });

    this.uniData = new ArrayBuffer(UNI_BYTES);
    this.uniF32  = new Float32Array(this.uniData);
    this.uniU32  = new Uint32Array(this.uniData);
    this.uniBuf  = device.createBuffer({
      size: UNI_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.stagingPosBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.stagingVelBuf = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.hostPos = new Float32Array(this.nBodies * 4);
    this.hostVel = new Float32Array(this.nBodies * 4);

    this.bboxBuf = device.createBuffer({
      size: 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Morton buffers are sized to the padded body count so bitonic sort can
    // treat the array as a power-of-two without index guards outside kernels.
    const mortonBytes = this.nBodiesPadded * 4;
    this.mortonKeysBuf = device.createBuffer({
      size: mortonBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.mortonIdxBuf = device.createBuffer({
      size: mortonBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // 2·N − 1 LBVH nodes, 64 bytes each. Degenerate-small case still needs at
    // least one node so the buffer-size validator is happy.
    const lbvhNodeCount = Math.max(2 * this.nBodies - 1, 1);
    this.lbvhNodesBuf = device.createBuffer({
      size: lbvhNodeCount * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Precompute bitonic stage sequence. For N padded to 2^p, there are
    // p(p+1)/2 stages — at most ≈210 for N = 2^20.
    const stages: Array<readonly [number, number]> = [];
    for (let k = 2; k <= this.nBodiesPadded; k *= 2) {
      for (let j = k / 2; j > 0; j /= 2) stages.push([k, j] as const);
    }
    this.sortStages = stages;

    this.sortParamsBuf = device.createBuffer({
      size: Math.max(stages.length, 1) * DYN_OFFSET_ALIGN,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Write stage params once at construction — they depend only on
    // nBodiesPadded, which doesn't change over the sim's lifetime.
    {
      const buf = new ArrayBuffer(stages.length * DYN_OFFSET_ALIGN);
      const dv  = new DataView(buf);
      for (let i = 0; i < stages.length; i++) {
        dv.setUint32(i * DYN_OFFSET_ALIGN + 0, stages[i][0], true);
        dv.setUint32(i * DYN_OFFSET_ALIGN + 4, stages[i][1], true);
      }
      device.queue.writeBuffer(this.sortParamsBuf, 0, buf);
    }

    const module = device.createShaderModule({ code: kernelSrc });
    this.bgLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.sortParamsLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 8 },
      }],
    });
    const plLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bgLayout],
    });
    const sortPlLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bgLayout, this.sortParamsLayout],
    });
    this.sortParamsBindGroup = device.createBindGroup({
      layout: this.sortParamsLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.sortParamsBuf, offset: 0, size: 8 },
      }],
    });

    this.forceDirectPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'forces_direct' },
    });
    this.forceLBVHPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'forces_lbvh' },
    });
    this.pnPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'pn_correction' },
    });
    this.kickPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'kick' },
    });
    this.driftPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'drift' },
    });
    this.bboxResetPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'bbox_reset' },
    });
    this.bboxReducePipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'bbox_reduce' },
    });
    this.mortonPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'compute_morton' },
    });
    this.sortPipeline = device.createComputePipeline({
      layout: sortPlLayout, compute: { module, entryPoint: 'bitonic_step' },
    });
    this.lbvhResetPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'lbvh_reset' },
    });
    this.buildHierarchyPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'build_hierarchy' },
    });
    this.summarizeLeavesPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'summarize_leaves' },
    });
    this.summarizeInternalPassPipeline = device.createComputePipeline({
      layout: plLayout, compute: { module, entryPoint: 'summarize_internal_pass' },
    });

    this.rebuildBindGroup();
  }

  destroy(): void {
    this.posBuf.destroy();
    this.velBuf.destroy();
    this.accBuf.destroy();
    this.uniBuf.destroy();
    this.stagingPosBuf.destroy();
    this.stagingVelBuf.destroy();
    this.bboxBuf.destroy();
    this.mortonKeysBuf.destroy();
    this.mortonIdxBuf.destroy();
    this.lbvhNodesBuf.destroy();
    this.sortParamsBuf.destroy();
  }

  private rebuildBindGroup(): void {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bgLayout,
      entries: [
        { binding: 0, resource: { buffer: this.posBuf } },
        { binding: 1, resource: { buffer: this.velBuf } },
        { binding: 2, resource: { buffer: this.accBuf } },
        { binding: 3, resource: { buffer: this.uniBuf } },
        { binding: 5, resource: { buffer: this.bboxBuf } },
        { binding: 6, resource: { buffer: this.mortonKeysBuf } },
        { binding: 7, resource: { buffer: this.mortonIdxBuf } },
        { binding: 8, resource: { buffer: this.lbvhNodesBuf } },
      ],
    });
  }

  setState(
    starPos:  Float32Array,
    starVel:  Float32Array,
    starMass: Float32Array,
    sinks:    SinkInit[],
  ): void {
    if (starPos.length  !== 3 * this.nStars) throw new Error('starPos size');
    if (starVel.length  !== 3 * this.nStars) throw new Error('starVel size');
    if (starMass.length !==     this.nStars) throw new Error('starMass size');
    if (sinks.length    !==     this.nSinks) throw new Error('sinks size');

    const n = this.nBodies;
    const pos = new Float32Array(n * 4);
    const vel = new Float32Array(n * 4);
    for (let i = 0; i < this.nStars; i++) {
      pos[4 * i + 0] = starPos[3 * i + 0];
      pos[4 * i + 1] = starPos[3 * i + 1];
      pos[4 * i + 2] = starPos[3 * i + 2];
      pos[4 * i + 3] = starMass[i];
      vel[4 * i + 0] = starVel[3 * i + 0];
      vel[4 * i + 1] = starVel[3 * i + 1];
      vel[4 * i + 2] = starVel[3 * i + 2];
    }
    for (let s = 0; s < this.nSinks; s++) {
      const i = this.nStars + s;
      const k = sinks[s];
      pos[4 * i + 0] = k.x;  pos[4 * i + 1] = k.y;  pos[4 * i + 2] = k.z;
      pos[4 * i + 3] = k.mass;
      vel[4 * i + 0] = k.vx; vel[4 * i + 1] = k.vy; vel[4 * i + 2] = k.vz;
    }

    this.device.queue.writeBuffer(this.posBuf, 0, pos);
    this.device.queue.writeBuffer(this.velBuf, 0, vel);
    const zero = new Float32Array(n * 4);
    this.device.queue.writeBuffer(this.accBuf, 0, zero);
    this.accelPrimed = false;

    // Seed the sink-state mirror from the init values directly.
    this.sinks = sinks.map((s) => ({ ...s }));
  }

  private writeConstants(): void {
    const p = this.params;
    const halos = p.halos ?? [];
    const nHalos = Math.min(halos.length, MAX_HALOS);

    this.uniF32[0] = p.softening * p.softening;
    this.uniF32[1] = p.G;
    this.uniF32[2] = p.dt;
    this.uniF32[3] = 0.5 * p.dt;
    this.uniF32[4] = p.c;
    this.uniF32[5] = p.pnAmp;
    this.uniF32[6] = p.theta * p.theta;
    this.uniF32[7] = 0;
    this.uniU32[8]  = this.nStars;
    this.uniU32[9]  = this.nSinks;
    this.uniU32[10] = this.nBodies;
    this.uniU32[11] = nHalos;
    // u32 index 52 is the n_bodies_padded field (after the 160-byte halos block
    // of 5 halos). Indices 53..55 are trailing struct pad bytes.
    this.uniU32[52] = this.nBodiesPadded;
    this.uniU32[53] = 0;
    this.uniU32[54] = 0;
    this.uniU32[55] = 0;

    for (let h = 0; h < MAX_HALOS; h++) {
      const base = 12 + h * 8;
      if (h < nHalos) {
        const halo = halos[h];
        this.uniF32[base + 0] = halo.cx;
        this.uniF32[base + 1] = halo.cy;
        this.uniF32[base + 2] = halo.cz;
        this.uniF32[base + 3] = halo.v0 * halo.v0;
        this.uniF32[base + 4] = halo.rc * halo.rc;
        this.uniF32[base + 5] = 0;
        this.uniF32[base + 6] = 0;
        this.uniF32[base + 7] = 0;
      } else {
        for (let k = 0; k < 8; k++) this.uniF32[base + k] = 0;
      }
    }

    this.device.queue.writeBuffer(this.uniBuf, 0, this.uniData);
  }

  private groupsFor(n: number): number {
    return Math.max(1, Math.ceil(n / WG_SIZE));
  }

  // Readback is only for sink-related CPU work (accretion + mirror). The
  // tree build is fully on the GPU now.
  private needsPrestepReadback(): boolean {
    return this.nSinks > 0;
  }

  private async readbackPosVel(): Promise<void> {
    const bytes = this.nBodies * 4 * Float32Array.BYTES_PER_ELEMENT;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.posBuf, 0, this.stagingPosBuf, 0, bytes);
    enc.copyBufferToBuffer(this.velBuf, 0, this.stagingVelBuf, 0, bytes);
    this.device.queue.submit([enc.finish()]);

    await Promise.all([
      this.stagingPosBuf.mapAsync(GPUMapMode.READ, 0, bytes),
      this.stagingVelBuf.mapAsync(GPUMapMode.READ, 0, bytes),
    ]);
    this.hostPos.set(new Float32Array(this.stagingPosBuf.getMappedRange(0, bytes)));
    this.hostVel.set(new Float32Array(this.stagingVelBuf.getMappedRange(0, bytes)));
    this.stagingPosBuf.unmap();
    this.stagingVelBuf.unmap();
  }

  // CPU accretion: stars within captureRadius of any sink are absorbed.
  // Their mass+momentum are merged into the sink using conservation.
  // Directly mirrors Simulation::Impl::doAccretion in simulation.mm.
  private runAccretion(): boolean {
    if (this.nSinks === 0 || this.nStars === 0) return false;
    const rCap = Math.max(this.params.captureRadius, 0);
    if (rCap <= 0) return false;
    const rCap2 = rCap * rCap;

    const pos = this.hostPos;
    const vel = this.hostVel;

    const dm  = new Float64Array(this.nSinks);
    const dpx = new Float64Array(this.nSinks);
    const dpy = new Float64Array(this.nSinks);
    const dpz = new Float64Array(this.nSinks);
    let captured = 0;

    for (let i = 0; i < this.nStars; i++) {
      const mi = pos[4 * i + 3];
      if (mi <= 0) continue;
      for (let s = 0; s < this.nSinks; s++) {
        const j = this.nStars + s;
        const dx = pos[4 * i + 0] - pos[4 * j + 0];
        const dy = pos[4 * i + 1] - pos[4 * j + 1];
        const dz = pos[4 * i + 2] - pos[4 * j + 2];
        if (dx * dx + dy * dy + dz * dz < rCap2) {
          dm[s]  += mi;
          dpx[s] += mi * vel[4 * i + 0];
          dpy[s] += mi * vel[4 * i + 1];
          dpz[s] += mi * vel[4 * i + 2];
          pos[4 * i + 3] = 0;
          captured++;
          break;
        }
      }
    }

    for (let s = 0; s < this.nSinks; s++) {
      if (dm[s] <= 0) continue;
      const j = this.nStars + s;
      const mOld = pos[4 * j + 3];
      const mNew = mOld + dm[s];
      vel[4 * j + 0] = (mOld * vel[4 * j + 0] + dpx[s]) / mNew;
      vel[4 * j + 1] = (mOld * vel[4 * j + 1] + dpy[s]) / mNew;
      vel[4 * j + 2] = (mOld * vel[4 * j + 2] + dpz[s]) / mNew;
      pos[4 * j + 3] = mNew;
    }

    return captured > 0;
  }

  private updateSinkMirror(): void {
    const out: SinkState[] = [];
    for (let s = 0; s < this.nSinks; s++) {
      const i = this.nStars + s;
      out.push({
        x:    this.hostPos[4 * i + 0],
        y:    this.hostPos[4 * i + 1],
        z:    this.hostPos[4 * i + 2],
        vx:   this.hostVel[4 * i + 0],
        vy:   this.hostVel[4 * i + 1],
        vz:   this.hostVel[4 * i + 2],
        mass: this.hostPos[4 * i + 3],
      });
    }
    this.sinks = out;
  }

  async stepMany(nSteps: number): Promise<void> {
    if (nSteps <= 0) return;
    this.writeConstants();

    const forcePipeline = this.params.useBarnesHut
      ? this.forceLBVHPipeline
      : this.forceDirectPipeline;

    let stateDirty = false;
    if (this.needsPrestepReadback()) {
      await this.readbackPosVel();
      stateDirty = this.runAccretion();
      this.updateSinkMirror();
      if (stateDirty) {
        this.device.queue.writeBuffer(this.posBuf, 0, this.hostPos);
        this.device.queue.writeBuffer(this.velBuf, 0, this.hostVel);
        this.accelPrimed = false;
      }
    }

    const groups       = this.groupsFor(this.nBodies);
    const groupsPadded = this.groupsFor(this.nBodiesPadded);
    const enc = this.device.createCommandEncoder();

    // GPU LBVH rebuild: bbox → morton → bitonic sort → Karras → summarize.
    // Cost scales as O(N) + O(N·log²N) for the sort. Replaces the CPU
    // tree build entirely.
    if (this.params.useBarnesHut) {
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.bindGroup);

      pass.setPipeline(this.bboxResetPipeline);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(this.bboxReducePipeline);
      pass.dispatchWorkgroups(groups);
      pass.setPipeline(this.mortonPipeline);
      pass.dispatchWorkgroups(groupsPadded);

      pass.setPipeline(this.sortPipeline);
      for (let i = 0; i < this.sortStages.length; i++) {
        pass.setBindGroup(1, this.sortParamsBindGroup, [i * DYN_OFFSET_ALIGN]);
        pass.dispatchWorkgroups(groupsPadded);
      }

      const totalNodes    = Math.max(2 * this.nBodies - 1, 1);
      const groupsInternal = this.groupsFor(Math.max(this.nBodies - 1, 1));
      pass.setPipeline(this.lbvhResetPipeline);
      pass.dispatchWorkgroups(this.groupsFor(totalNodes));
      pass.setPipeline(this.buildHierarchyPipeline);
      pass.dispatchWorkgroups(groupsInternal);
      pass.setPipeline(this.summarizeLeavesPipeline);
      pass.dispatchWorkgroups(groups);
      // Upper bound on any LBVH's depth (worst case 2·log₂N); add slack so we
      // still converge on clustered inputs.
      const summarizePasses = Math.max(1, 2 * Math.ceil(Math.log2(Math.max(this.nBodies, 2))) + 4);
      pass.setPipeline(this.summarizeInternalPassPipeline);
      for (let i = 0; i < summarizePasses; i++) {
        pass.dispatchWorkgroups(groupsInternal);
      }
      pass.end();
    }

    if (!this.accelPrimed) {
      const pp = enc.beginComputePass();
      pp.setBindGroup(0, this.bindGroup);
      pp.setPipeline(forcePipeline);
      pp.dispatchWorkgroups(groups);
      if (this.nSinks > 0 && this.params.pnAmp !== 0) {
        pp.setPipeline(this.pnPipeline);
        pp.dispatchWorkgroups(1);
      }
      pp.end();
      this.accelPrimed = true;
    }

    const runPN = this.nSinks > 0 && this.params.pnAmp !== 0;
    const p = enc.beginComputePass();
    p.setBindGroup(0, this.bindGroup);
    for (let s = 0; s < nSteps; s++) {
      p.setPipeline(this.kickPipeline);
      p.dispatchWorkgroups(groups);
      p.setPipeline(this.driftPipeline);
      p.dispatchWorkgroups(groups);
      p.setPipeline(forcePipeline);
      p.dispatchWorkgroups(groups);
      if (runPN) {
        p.setPipeline(this.pnPipeline);
        p.dispatchWorkgroups(1);
      }
      p.setPipeline(this.kickPipeline);
      p.dispatchWorkgroups(groups);
    }
    p.end();

    this.device.queue.submit([enc.finish()]);
  }
}
