// A/B harness for the smoothed-blend-normal fix in displacement.js.
//
// Hypothesis: on noisy/sculpted meshes (like the bear), high-frequency
// surface-normal jitter amplifies through the cubic / triplanar blend
// because two unrelated heightmap samples are mixed at each vertex by a
// noisy weight. Smoothing the BLEND normal (not the displacement direction)
// damps that amplification at the source. On a smooth sphere the smoothing
// should be a near-no-op.
//
// We run applyDisplacement twice — blendNormalSmoothing=0 vs blendNormalSmoothing=4 —
// on the same mesh + heightmap, then compare per-vertex displacement noise.
//
// Roughness metric: |Laplacian| of the per-unique-vertex displacement scalar
// across the dedup-graph (each vertex vs. mean of its neighbours). Reported
// for the whole surface AND for the "blend zone" subset (vertices whose
// triplanar weight is not dominated by one axis), since that's where the
// fix is supposed to matter.
//
// Run: node test-blend-noise.mjs

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { applyDisplacement } from './js/displacement.js';
import { MODE_TRIPLANAR, MODE_CUBIC } from './js/mapping.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Deterministic PRNG (mulberry32) ──────────────────────────────────────
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Mesh synthesis: UV-sphere with optional radial noise ──────────────────
// `noise` = max relative radial perturbation. 0 → smooth sphere; 0.04 → bumpy
// sculpted-style surface (the bear-noise stand-in).
function makeBumpySphere(radius, lat, lon, noise, seed) {
  const rand = mulberry32(seed);
  const verts = new Float32Array((lat + 1) * (lon + 1) * 3);
  const cols = lon + 1;
  for (let i = 0; i <= lat; i++) {
    const phi = (i / lat) * Math.PI;
    const sp = Math.sin(phi), cp = Math.cos(phi);
    for (let j = 0; j <= lon; j++) {
      const theta = (j / lon) * 2 * Math.PI;
      const r = radius * (1 + (noise > 0 ? (rand() - 0.5) * 2 * noise : 0));
      const idx = (i * cols + j) * 3;
      verts[idx]     = sp * Math.cos(theta) * r;
      verts[idx + 1] = cp * r;
      verts[idx + 2] = sp * Math.sin(theta) * r;
    }
  }

  const triCount = lat * lon * 2;
  const positions = new Float32Array(triCount * 9);
  let p = 0;
  for (let i = 0; i < lat; i++) {
    for (let j = 0; j < lon; j++) {
      const a = (i * cols + j) * 3;
      const b = ((i + 1) * cols + j) * 3;
      const c = ((i + 1) * cols + j + 1) * 3;
      const d = (i * cols + j + 1) * 3;
      // tri 1: a, b, c
      positions[p++] = verts[a];   positions[p++] = verts[a+1]; positions[p++] = verts[a+2];
      positions[p++] = verts[b];   positions[p++] = verts[b+1]; positions[p++] = verts[b+2];
      positions[p++] = verts[c];   positions[p++] = verts[c+1]; positions[p++] = verts[c+2];
      // tri 2: a, c, d
      positions[p++] = verts[a];   positions[p++] = verts[a+1]; positions[p++] = verts[a+2];
      positions[p++] = verts[c];   positions[p++] = verts[c+1]; positions[p++] = verts[c+2];
      positions[p++] = verts[d];   positions[p++] = verts[d+1]; positions[p++] = verts[d+2];
    }
  }

  // Per-face normals duplicated to all 3 vertex copies (faceted) — but we
  // also need *interpolated* normals so displacement.js's "use buffer normal"
  // path produces smooth normals. Use computeVertexNormals on an indexed
  // version, then unroll.
  const indexed = new THREE.BufferGeometry();
  // Build an indexed sphere from the same verts grid (no dedup needed —
  // verts grid is already unique since we built it that way).
  const indexArr = new Uint32Array(triCount * 3);
  let ip = 0;
  for (let i = 0; i < lat; i++) {
    for (let j = 0; j < lon; j++) {
      const a = i * cols + j;
      const b = (i + 1) * cols + j;
      const c = (i + 1) * cols + j + 1;
      const d = i * cols + j + 1;
      indexArr[ip++] = a; indexArr[ip++] = b; indexArr[ip++] = c;
      indexArr[ip++] = a; indexArr[ip++] = c; indexArr[ip++] = d;
    }
  }
  indexed.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  indexed.setIndex(new THREE.BufferAttribute(indexArr, 1));
  indexed.computeVertexNormals();

  // Unroll to non-indexed with interpolated normals
  const normals = new Float32Array(triCount * 9);
  const indexedNrm = indexed.attributes.normal.array;
  for (let t = 0; t < triCount; t++) {
    for (let v = 0; v < 3; v++) {
      const sourceIdx = indexArr[t * 3 + v];
      normals[t * 9 + v * 3]     = indexedNrm[sourceIdx * 3];
      normals[t * 9 + v * 3 + 1] = indexedNrm[sourceIdx * 3 + 1];
      normals[t * 9 + v * 3 + 2] = indexedNrm[sourceIdx * 3 + 2];
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('normal',   new THREE.BufferAttribute(normals,  3));
  return geo;
}

// ── Structured heightmap (box-filtered noise) ────────────────────────────
// We need adjacent samples on the SAME projection axis to be correlated, so
// roughness in the dominant zones comes only from texture detail (smooth)
// while roughness in the blend zone comes from mixing two unrelated UV
// regions. Pure white noise hides the signal because the dominant zones are
// already maximally noisy (adjacent pixels uncorrelated).
function makeNoiseHeightmap(w, h, seed, blurPasses = 4) {
  const rand = mulberry32(seed);
  let plane = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) plane[i] = rand();
  // Repeated 3-tap box filter (separable, wrapped) — turns white noise into
  // smooth blobs. After ~4 passes the spectrum is dominated by low frequencies.
  const tmp = new Float32Array(w * h);
  for (let pass = 0; pass < blurPasses; pass++) {
    // Horizontal
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const xm = (x - 1 + w) % w;
        const xp = (x + 1) % w;
        tmp[y*w + x] = (plane[y*w + xm] + plane[y*w + x] + plane[y*w + xp]) / 3;
      }
    }
    // Vertical
    for (let y = 0; y < h; y++) {
      const ym = (y - 1 + h) % h;
      const yp = (y + 1) % h;
      for (let x = 0; x < w; x++) {
        plane[y*w + x] = (tmp[ym*w + x] + tmp[y*w + x] + tmp[yp*w + x]) / 3;
      }
    }
  }
  // Normalise to [0,1] and pack to RGBA8
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < w*h; i++) { if (plane[i] < mn) mn = plane[i]; if (plane[i] > mx) mx = plane[i]; }
  const span = mx - mn || 1;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = Math.floor(((plane[i] - mn) / span) * 255);
    data[i*4] = v; data[i*4+1] = v; data[i*4+2] = v; data[i*4+3] = 255;
  }
  return { data, width: w, height: h };
}

function computeBounds(geo) {
  geo.computeBoundingBox();
  const bb = geo.boundingBox;
  return {
    min: bb.min.clone(),
    max: bb.max.clone(),
    size: bb.max.clone().sub(bb.min),
    center: bb.max.clone().add(bb.min).multiplyScalar(0.5),
  };
}

// ── Per-unique-vertex displacement scalar ─────────────────────────────────
// Mirrors the dedup that displacement.js does internally so we can attach
// per-vertex outputs to a single id.
function dedupAndExtract(inGeo, outGeo) {
  const inPos = inGeo.attributes.position;
  const inNrm = inGeo.attributes.normal;
  const outPos = outGeo.attributes.position;
  const count = inPos.count;
  const QUANT = 1e5;
  const dedupMap = new Map();
  let nextId = 0;
  const vertexId = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const x = inPos.getX(i), y = inPos.getY(i), z = inPos.getZ(i);
    const key = `${Math.round(x*QUANT)}_${Math.round(y*QUANT)}_${Math.round(z*QUANT)}`;
    let id = dedupMap.get(key);
    if (id === undefined) { id = nextId++; dedupMap.set(key, id); }
    vertexId[i] = id;
  }
  const uniqueCount = nextId;
  const dispMag = new Float64Array(uniqueCount);
  const seen = new Uint8Array(uniqueCount);
  for (let i = 0; i < count; i++) {
    const vid = vertexId[i];
    if (seen[vid]) continue;
    seen[vid] = 1;
    const dx = outPos.getX(i) - inPos.getX(i);
    const dy = outPos.getY(i) - inPos.getY(i);
    const dz = outPos.getZ(i) - inPos.getZ(i);
    dispMag[vid] = dx * inNrm.getX(i) + dy * inNrm.getY(i) + dz * inNrm.getZ(i);
  }
  return { vertexId, uniqueCount, dispMag };
}

function buildAdjacency(vertexId, count, uniqueCount) {
  const degree = new Uint32Array(uniqueCount);
  for (let t = 0; t < count; t += 3) {
    const a = vertexId[t], b = vertexId[t+1], c = vertexId[t+2];
    if (a !== b) { degree[a]++; degree[b]++; }
    if (b !== c) { degree[b]++; degree[c]++; }
    if (c !== a) { degree[c]++; degree[a]++; }
  }
  const csrStart = new Uint32Array(uniqueCount + 1);
  for (let i = 0; i < uniqueCount; i++) csrStart[i+1] = csrStart[i] + degree[i];
  const neighbors = new Uint32Array(csrStart[uniqueCount]);
  const cursor = new Uint32Array(uniqueCount);
  for (let t = 0; t < count; t += 3) {
    const a = vertexId[t], b = vertexId[t+1], c = vertexId[t+2];
    if (a !== b) { neighbors[csrStart[a] + cursor[a]++] = b; neighbors[csrStart[b] + cursor[b]++] = a; }
    if (b !== c) { neighbors[csrStart[b] + cursor[b]++] = c; neighbors[csrStart[c] + cursor[c]++] = b; }
    if (c !== a) { neighbors[csrStart[c] + cursor[c]++] = a; neighbors[csrStart[a] + cursor[a]++] = c; }
  }
  return { csrStart, neighbors };
}

// Per-unique-vertex average input normal — used to classify "blend zone".
function avgUniqueNormal(inGeo, vertexId, count, uniqueCount) {
  const nrm = inGeo.attributes.normal;
  const sumX = new Float64Array(uniqueCount);
  const sumY = new Float64Array(uniqueCount);
  const sumZ = new Float64Array(uniqueCount);
  for (let i = 0; i < count; i++) {
    const id = vertexId[i];
    sumX[id] += nrm.getX(i);
    sumY[id] += nrm.getY(i);
    sumZ[id] += nrm.getZ(i);
  }
  for (let id = 0; id < uniqueCount; id++) {
    const len = Math.sqrt(sumX[id]**2 + sumY[id]**2 + sumZ[id]**2) || 1;
    sumX[id] /= len; sumY[id] /= len; sumZ[id] /= len;
  }
  return { nx: sumX, ny: sumY, nz: sumZ };
}

// Roughness = mean |dispMag - mean(dispMag at neighbours)|.
// Filtered by triplanar dominant-axis weight (a^4 normalised); a vertex is
// in the "blend zone" if no axis dominates above wThresh.
function roughness(dispMag, csr, normals, wThresh = 0.85) {
  const u = dispMag.length;
  let sumAll = 0, nAll = 0;
  let sumBlend = 0, nBlend = 0;
  for (let id = 0; id < u; id++) {
    const s = csr.csrStart[id], e = csr.csrStart[id+1];
    if (e === s) continue;
    let mean = 0;
    for (let k = s; k < e; k++) mean += dispMag[csr.neighbors[k]];
    mean /= (e - s);
    const aL = Math.abs(dispMag[id] - mean);
    sumAll += aL; nAll++;

    const ax = Math.abs(normals.nx[id]);
    const ay = Math.abs(normals.ny[id]);
    const az = Math.abs(normals.nz[id]);
    const bx = ax*ax*ax*ax, by = ay*ay*ay*ay, bz = az*az*az*az;
    const wMax = Math.max(bx, by, bz) / (bx + by + bz + 1e-9);
    if (wMax < wThresh) { sumBlend += aL; nBlend++; }
  }
  return {
    overall: nAll > 0 ? sumAll / nAll : 0,
    blendZone: nBlend > 0 ? sumBlend / nBlend : 0,
    blendCount: nBlend,
    totalCount: nAll,
  };
}

// ── Run the test ──────────────────────────────────────────────────────────
console.log('=== Texture-blend seam noise: smoothed-blend-normal A/B test ===\n');

const radius = 50;
const heightmap = makeNoiseHeightmap(256, 256, 7);
const baseSettings = {
  scaleU: 0.4, scaleV: 0.4, offsetU: 0, offsetV: 0, rotation: 0,
  amplitude: 1.0,
  symmetricDisplacement: true,  // centred so disp is signed around 0
  bottomAngleLimit: 0, topAngleLimit: 0,
  capAngle: 20,
  noDownwardZ: false,
};

// Load the filleted cube (fillets create localized blend zones at the edges).
function loadBinarySTL(filePath) {
  const buf = fs.readFileSync(filePath);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const triCount = dv.getUint32(80, true);
  const positions = new Float32Array(triCount * 9);
  const normals   = new Float32Array(triCount * 9);
  for (let i = 0; i < triCount; i++) {
    const base = 84 + i * 50;
    const nx = dv.getFloat32(base,     true);
    const ny = dv.getFloat32(base + 4, true);
    const nz = dv.getFloat32(base + 8, true);
    for (let v = 0; v < 3; v++) {
      const off = base + 12 + v * 12;
      positions[i*9 + v*3]     = dv.getFloat32(off,     true);
      positions[i*9 + v*3 + 1] = dv.getFloat32(off + 4, true);
      positions[i*9 + v*3 + 2] = dv.getFloat32(off + 8, true);
      normals[i*9 + v*3]     = nx;
      normals[i*9 + v*3 + 1] = ny;
      normals[i*9 + v*3 + 2] = nz;
    }
  }
  // Recompute interpolated (smooth) normals via indexed pass — STL only stores
  // face normals.
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

// Add radial / arbitrary-direction position noise to an existing mesh (mimics
// bear-style organic surface noise). Renormalises smooth normals afterwards.
function addPositionNoise(geo, amplitude, seed) {
  const rand = mulberry32(seed);
  const pos = geo.attributes.position;
  const count = pos.count;
  // Dedup so we move shared vertices coherently (no cracks).
  const QUANT = 1e5;
  const dedup = new Map();
  let nextId = 0;
  const vid = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const k = `${Math.round(pos.getX(i)*QUANT)}_${Math.round(pos.getY(i)*QUANT)}_${Math.round(pos.getZ(i)*QUANT)}`;
    let id = dedup.get(k);
    if (id === undefined) { id = nextId++; dedup.set(k, id); }
    vid[i] = id;
  }
  const offX = new Float32Array(nextId);
  const offY = new Float32Array(nextId);
  const offZ = new Float32Array(nextId);
  for (let id = 0; id < nextId; id++) {
    offX[id] = (rand() - 0.5) * 2 * amplitude;
    offY[id] = (rand() - 0.5) * 2 * amplitude;
    offZ[id] = (rand() - 0.5) * 2 * amplitude;
  }
  const arr = pos.array;
  for (let i = 0; i < count; i++) {
    arr[i*3]     += offX[vid[i]];
    arr[i*3 + 1] += offY[vid[i]];
    arr[i*3 + 2] += offZ[vid[i]];
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

const cubePath = path.join(__dirname, 'cubeWithSmallFillets.stl');
const cubeGeo = loadBinarySTL(cubePath);
const noisyCubeGeo = addPositionNoise(loadBinarySTL(cubePath), 0.08, 3);

const meshes = [
  { label: 'smooth sphere (control)',           geo: makeBumpySphere(radius, 64, 96, 0.00, 1) },
  { label: 'bumpy sphere (4% radial noise)',    geo: makeBumpySphere(radius, 64, 96, 0.04, 2) },
  { label: 'filleted cube (smooth)',            geo: cubeGeo },
  { label: 'filleted cube + 0.08mm noise',      geo: noisyCubeGeo },
];

const modes = [
  { label: 'TRIPLANAR',     mode: MODE_TRIPLANAR, mappingBlend: 0,    seamBandWidth: 0.5  },
  { label: 'CUBIC blend=1', mode: MODE_CUBIC,     mappingBlend: 1.0,  seamBandWidth: 0.35 },
];

const smoothingLevels = [0, 2, 4, 8];

for (const { label, geo } of meshes) {
  const triCount = geo.attributes.position.count / 3;
  console.log(`— ${label} (${triCount.toLocaleString()} triangles) —`);

  for (const m of modes) {
    const settings = {
      ...baseSettings,
      mappingMode: m.mode,
      mappingBlend: m.mappingBlend,
      seamBandWidth: m.seamBandWidth,
    };
    const bounds = computeBounds(geo);

    const results = [];
    let baselineDisp = null;
    for (const k of smoothingLevels) {
      const out = applyDisplacement(geo, heightmap, heightmap.width, heightmap.height,
        { ...settings, blendNormalSmoothing: k }, bounds);
      const { vertexId, uniqueCount, dispMag } = dedupAndExtract(geo, out);
      const adj = buildAdjacency(vertexId, geo.attributes.position.count, uniqueCount);
      const normals = avgUniqueNormal(geo, vertexId, geo.attributes.position.count, uniqueCount);
      const r = roughness(dispMag, adj, normals);
      // How much did per-vertex displacement actually change vs k=0?
      let meanAbsChange = 0;
      if (baselineDisp) {
        let s = 0;
        for (let id = 0; id < uniqueCount; id++) s += Math.abs(dispMag[id] - baselineDisp[id]);
        meanAbsChange = s / uniqueCount;
      } else {
        baselineDisp = new Float64Array(dispMag);
      }
      results.push({ k, ...r, meanAbsChange });
    }

    console.log(`  ${m.label}:`);
    const r0 = results[0];
    console.log(`    blend zone: ${r0.blendCount}/${r0.totalCount} unique vertices`);
    console.log(`    smoothing | overall roughness | blend-zone roughness | blend Δ vs k=0 | mean|Δh| vs k=0`);
    for (const r of results) {
      const dPct = r0.blendZone > 0 ? ((r.blendZone / r0.blendZone - 1) * 100).toFixed(1) : '—';
      console.log(`         k=${r.k}  |  ${r.overall.toExponential(3).padStart(11)}  |  ${r.blendZone.toExponential(3).padStart(11)}  |  ${dPct.padStart(6)}%        |  ${r.meanAbsChange.toExponential(3)}`);
    }
    console.log();
  }
}
