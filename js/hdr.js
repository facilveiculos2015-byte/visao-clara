// HDR/EDR presenter. Tries, in order:
//  1) WebGPU canvas: format rgba16float + toneMapping {mode:'extended'} (Safari 26+/Chrome where supported)
//  2) WebGL2 extended canvas (drawingBufferStorage RGBA16F + configureHighDynamicRange) – handled in gpu.js
// The web does not expose the current EDR headroom, so the box upper bound H is a fixed, user-adjustable value.
// Values written are *extended sRGB encoded* (sign-preserving OETF); 1.0 = SDR white.
export class WebGPUPresenter {
  static async create(N) {
    const why = [];
    if (!navigator.gpu) return { ok: false, reason: "WebGPU indisponível" };
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { ok: false, reason: "sem adaptador WebGPU" };
      const device = await adapter.requestDevice();
      const canvas = document.createElement("canvas"); canvas.width = canvas.height = N;
      const ctx = canvas.getContext("webgpu");
      ctx.configure({ device, format: "rgba16float", colorSpace: "srgb", toneMapping: { mode: "extended" }, alphaMode: "opaque" });
      const cfg = ctx.getConfiguration ? ctx.getConfiguration() : null;
      const tm = cfg && cfg.toneMapping ? cfg.toneMapping.mode : "desconhecido";
      if (tm !== "extended") return { ok: false, reason: `WebGPU sem tone mapping 'extended' (${tm})` };
      const highScreen = matchMedia("(dynamic-range: high)").matches;
      const p = new WebGPUPresenter(device, ctx, canvas, N);
      return { ok: true, presenter: p, reason: highScreen ? "WebGPU extended + tela HDR" : "WebGPU extended (tela não reporta HDR — pode não haver folga real)" };
    } catch (e) { return { ok: false, reason: "WebGPU falhou: " + e }; }
  }
  constructor(device, ctx, canvas, N) {
    this.device = device; this.ctx = ctx; this.canvas = canvas; this.N = N;
    this.tex = device.createTexture({ size: [N, N], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const mod = device.createShaderModule({ code: `
      @group(0) @binding(0) var t: texture_2d<f32>;
      @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
        var p = array<vec2f,3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3)); return vec4f(p[i], 0, 1); }
      @fragment fn fs(@builtin(position) q: vec4f) -> @location(0) vec4f {
        let n = i32(${N});   // readPixels rows are bottom-first -> flip
        return textureLoad(t, vec2i(i32(q.x), n - 1 - i32(q.y)), 0); }` });
    this.pipe = device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
      fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba16float" }] }, primitive: { topology: "triangle-list" } });
    this.bg = device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.tex.createView() }] });
  }
  present(f32) {
    const d = this.device;
    d.queue.writeTexture({ texture: this.tex }, f32, { bytesPerRow: this.N * 16 }, [this.N, this.N]);
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
    pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg); pass.draw(3); pass.end();
    d.queue.submit([enc.finish()]);
  }
}
