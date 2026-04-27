import shaderSrc from '../shaders/render.wgsl?raw';
import type { SinkState } from './simulation.ts';

const MAX_SINKS = 5;
// Uniforms layout (matches render.wgsl struct Uniforms):
//   mat4x4 viewProj           at 0
//   vec2   pointNdc           at 64
//   vec2   sinkPointNdc       at 72
//   u32    perGalaxy          at 80
//   u32    galaxyCount        at 84
//   u32    nSinks             at 88
//   u32    _pad               at 92
//   vec4   sinks[5]           at 96   (16 bytes × 5 = 80 bytes)
const UNI_BYTES = 176;
const STAR_POINT_SIZE_PX = 2.5;
const SINK_POINT_SIZE_PX = 64.0;

export interface RenderUpdate {
  viewProj:    Float32Array;  // 16 floats, column-major
  widthPx:     number;
  heightPx:    number;
  perGalaxy:   number;        // stars per galaxy; shader picks palette slot by ii / perGalaxy
  galaxyCount: number;        // 1..5
  sinks:       SinkState[];
  G:           number;        // needed for Schwarzschild radius rs = 2GM/c²
  c:           number;
}

export interface Renderer {
  render(view: GPUTextureView, u: RenderUpdate): void;
  setNStars(n: number): void;
  setNSinks(n: number): void;
}

export function createRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  posBuf: GPUBuffer,
  nStarsInit: number,
  nSinksInit: number,
): Renderer {
  const module = device.createShaderModule({ code: shaderSrc });

  const bindLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const plLayout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });

  const blend: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };

  const starPipeline = device.createRenderPipeline({
    layout: plLayout,
    vertex:   { module, entryPoint: 'point_vs' },
    fragment: { module, entryPoint: 'point_fs',
                targets: [{ format, blend }] },
    primitive: { topology: 'triangle-list' },
  });

  const sinkPipeline = device.createRenderPipeline({
    layout: plLayout,
    vertex:   { module, entryPoint: 'sink_vs' },
    fragment: { module, entryPoint: 'sink_fs',
                targets: [{ format, blend }] },
    primitive: { topology: 'triangle-list' },
  });

  const uniBuf = device.createBuffer({
    size: UNI_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniData = new ArrayBuffer(UNI_BYTES);
  const uniF32  = new Float32Array(uniData);
  const uniU32  = new Uint32Array(uniData);

  const bindGroup = device.createBindGroup({
    layout: bindLayout,
    entries: [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: { buffer: posBuf } },
    ],
  });

  let nStars = nStarsInit;
  let nSinks = nSinksInit;

  return {
    setNStars(n) { nStars = n; },
    setNSinks(n) { nSinks = n; },
    render(view, u) {
      uniF32.set(u.viewProj, 0);
      // pointNdc / sinkPointNdc: convert pixel size to NDC extent.
      uniF32[16] = (2 * STAR_POINT_SIZE_PX) / u.widthPx;
      uniF32[17] = (2 * STAR_POINT_SIZE_PX) / u.heightPx;
      uniF32[18] = (2 * SINK_POINT_SIZE_PX) / u.widthPx;
      uniF32[19] = (2 * SINK_POINT_SIZE_PX) / u.heightPx;
      uniU32[20] = Math.max(1, u.perGalaxy >>> 0);
      uniU32[21] = Math.max(1, Math.min(u.galaxyCount, MAX_SINKS)) >>> 0;
      const nSinksUsed = Math.min(u.sinks.length, MAX_SINKS);
      uniU32[22] = nSinksUsed;
      uniU32[23] = 0;

      const c2 = Math.max(u.c * u.c, 1e-12);
      for (let s = 0; s < MAX_SINKS; s++) {
        const base = 24 + s * 4;
        if (s < nSinksUsed) {
          const sink = u.sinks[s];
          const rs = 2 * u.G * Math.max(sink.mass, 0) / c2;
          uniF32[base + 0] = sink.x;
          uniF32[base + 1] = sink.y;
          uniF32[base + 2] = sink.z;
          uniF32[base + 3] = rs;
        } else {
          uniF32[base + 0] = 0;
          uniF32[base + 1] = 0;
          uniF32[base + 2] = 0;
          uniF32[base + 3] = 0;
        }
      }
      device.queue.writeBuffer(uniBuf, 0, uniData);

      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view,
          clearValue: { r: 0.0, g: 0.0, b: 0.015, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setBindGroup(0, bindGroup);

      pass.setPipeline(starPipeline);
      pass.draw(6, nStars, 0, 0);

      if (nSinks > 0) {
        pass.setPipeline(sinkPipeline);
        pass.draw(6, nSinks, 0, nStars);
      }

      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}
