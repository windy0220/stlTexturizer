// Regression harness for smart resolution.
//
// Verifies:
//   1. analyzeTexture() classifies synthetic textures as expected
//      (smooth gradient → coarse PPE, sharp checker → fine PPE).
//   2. computeSurfaceArea() gives sane values on unit cube + laserPlate.
//   3. computeSmartResolution() picks a smaller edge for sharp textures than
//      for smooth textures on the same model + scale.
//   4. Estimated triangle count stays under the hybrid triangle budget.
//   5. The budget floor kicks in when maxTriangles is small + texture is fine.
//
// Run: node --max-old-space-size=8192 test-smart-resolution.mjs

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';

import { analyzeTexture } from './js/textureAnalysis.js';
import { computeSurfaceArea, computeBounds } from './js/stlLoader.js';
import { computeSmartResolution } from './js/smartResolution.js';
import {
  MODE_PLANAR_XY, MODE_TRIPLANAR,
} from './js/mapping.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STL_PATH  = path.join(__dirname, 'laserPlate.stl');

let _failed = 0;
function expect(label, ok, detail) {
  if (ok) console.log(`  PASS  ${label}`);
  else { console.log(`  FAIL  ${label}${detail ? ` :: ${detail}` : ''}`); _failed++; }
}
function section(name) { console.log(`\n— ${name} —`); }

// ── STL helpers ──────────────────────────────────────────────────────────────
function parseBinarySTL(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const triCount = dv.getUint32(80, true);
  const positions = new Float32Array(triCount * 9);
  for (let i = 0; i < triCount; i++) {
    const base = 84 + i * 50;
    for (let v = 0; v < 3; v++) {
      const off = base + 12 + v * 12;
      positions[i*9 + v*3]     = dv.getFloat32(off,     true);
      positions[i*9 + v*3 + 1] = dv.getFloat32(off + 4, true);
      positions[i*9 + v*3 + 2] = dv.getFloat32(off + 8, true);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

function makeCubeGeometry(side = 1) {
  // 12 triangles spanning [0,side]^3.
  const s = side;
  const v = [
    [0,0,0],[s,0,0],[s,s,0],[0,s,0], // z=0
    [0,0,s],[s,0,s],[s,s,s],[0,s,s], // z=1
  ];
  const tris = [
    [0,2,1],[0,3,2],          // bottom (z=0)
    [4,5,6],[4,6,7],          // top    (z=1)
    [0,1,5],[0,5,4],          // y=0
    [2,3,7],[2,7,6],          // y=1
    [1,2,6],[1,6,5],          // x=1
    [0,4,7],[0,7,3],          // x=0
  ];
  const positions = new Float32Array(tris.length * 9);
  let p = 0;
  for (const t of tris) for (const idx of t) {
    positions[p++] = v[idx][0]; positions[p++] = v[idx][1]; positions[p++] = v[idx][2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

// ── Synthetic textures ───────────────────────────────────────────────────────
// Build ImageData-shaped objects: { width, height, data: Uint8ClampedArray }.
// analyzeTexture only reads .width, .height, .data — it doesn't construct an
// actual ImageData, so this works in node without canvas.

function makeSmoothGradient(size = 512) {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const v = (x / (size - 1)) * 255; // linear ramp 0→255 in X
      data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 255;
    }
  }
  return { width: size, height: size, data };
}

function makeMidFreqSine(size = 512, periodPx = 32) {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const v = 128 + 100 * Math.sin((2 * Math.PI * x) / periodPx);
      data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 255;
    }
  }
  return { width: size, height: size, data };
}

function makeHardChecker(size = 512, cellPx = 8) {
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const cell = ((Math.floor(x / cellPx) + Math.floor(y / cellPx)) & 1);
      const v = cell ? 255 : 0;
      data[i] = v; data[i+1] = v; data[i+2] = v; data[i+3] = 255;
    }
  }
  return { width: size, height: size, data };
}

// ── Tests ────────────────────────────────────────────────────────────────────

section('analyzeTexture');
const smooth   = analyzeTexture(makeSmoothGradient());
const midFreq  = analyzeTexture(makeMidFreqSine());
const checker  = analyzeTexture(makeHardChecker());
console.log(`  smooth  : meanGrad=${smooth.meanGrad.toFixed(2)}  sharpFrac=${smooth.sharpFrac.toFixed(3)}  PPE=${smooth.pixelsPerEdge}`);
console.log(`  midFreq : meanGrad=${midFreq.meanGrad.toFixed(2)}  sharpFrac=${midFreq.sharpFrac.toFixed(3)}  PPE=${midFreq.pixelsPerEdge}`);
console.log(`  checker : meanGrad=${checker.meanGrad.toFixed(2)}  sharpFrac=${checker.sharpFrac.toFixed(3)}  PPE=${checker.pixelsPerEdge}`);
expect('smooth gradient → PPE=4.0', smooth.pixelsPerEdge === 4.0,
        `got ${smooth.pixelsPerEdge}`);
expect('mid-frequency sine → PPE in [1.0, 2.5]',
        midFreq.pixelsPerEdge >= 1.0 && midFreq.pixelsPerEdge <= 2.5,
        `got ${midFreq.pixelsPerEdge}`);
expect('hard checker → PPE=1.0', checker.pixelsPerEdge === 1.0,
        `got ${checker.pixelsPerEdge}`);
expect('checker has higher sharpFrac than smooth',
        checker.sharpFrac > smooth.sharpFrac);

section('analyzeTexture caching');
const td = makeHardChecker(64, 4);
const a = analyzeTexture(td);
const b = analyzeTexture(td);
expect('memoised result is the same object', a === b);

section('computeSurfaceArea');
const unitCube = makeCubeGeometry(1);
const cubeArea = computeSurfaceArea(unitCube);
expect('unit-cube surface area ≈ 6.0', Math.abs(cubeArea - 6.0) < 1e-5,
        `got ${cubeArea}`);

let plate = null;
if (fs.existsSync(STL_PATH)) {
  plate = parseBinarySTL(fs.readFileSync(STL_PATH));
  const plateArea = computeSurfaceArea(plate);
  // 65×65×2 plate with through-hole: outer faces ~ 2×65² + 4×65×2 = 8450 + 520 = 8970.
  // Hole removes ~2×(πr²) and adds inner cylinder; final stays in 7000–10000 range.
  console.log(`  laserPlate.stl area = ${plateArea.toFixed(1)} mm²`);
  expect('laserPlate area in plausible range [5000, 12000]',
          plateArea > 5000 && plateArea < 12000, `got ${plateArea}`);
} else {
  console.log('  laserPlate.stl missing — skipping plate-area check');
}

section('computeSmartResolution — cube + textures');
// 100 mm cube: detail edges separate cleanly above the 0.05 mm floor.
const cube = makeCubeGeometry(100);
const cubeBounds = computeBounds(cube);
const baseSettings = {
  mappingMode:    MODE_TRIPLANAR,
  scaleU:         0.5,
  scaleV:         0.5,
  textureAspectU: 1,
  textureAspectV: 1,
  maxTriangles:   750_000,
};
const tx_smooth  = { imageData: makeSmoothGradient() };
const tx_mid     = { imageData: makeMidFreqSine() };
const tx_checker = { imageData: makeHardChecker() };

const r_smooth  = computeSmartResolution({ geometry: cube, bounds: cubeBounds, settings: baseSettings, texture: tx_smooth });
const r_mid     = computeSmartResolution({ geometry: cube, bounds: cubeBounds, settings: baseSettings, texture: tx_mid });
const r_checker = computeSmartResolution({ geometry: cube, bounds: cubeBounds, settings: baseSettings, texture: tx_checker });

console.log(`  smooth  : edge=${r_smooth.edge}  detailEdge=${r_smooth.diagnostics.detailEdge.toFixed(4)}  budgetEdge=${r_smooth.diagnostics.budgetEdge.toFixed(4)}  estTris=${Math.round(r_smooth.diagnostics.estTriangles)}`);
console.log(`  mid     : edge=${r_mid.edge}  detailEdge=${r_mid.diagnostics.detailEdge.toFixed(4)}  budgetEdge=${r_mid.diagnostics.budgetEdge.toFixed(4)}  estTris=${Math.round(r_mid.diagnostics.estTriangles)}`);
console.log(`  checker : edge=${r_checker.edge}  detailEdge=${r_checker.diagnostics.detailEdge.toFixed(4)}  budgetEdge=${r_checker.diagnostics.budgetEdge.toFixed(4)}  estTris=${Math.round(r_checker.diagnostics.estTriangles)}`);

expect('checker gets a finer (≤) edge than smooth on same model',
        r_checker.edge <= r_smooth.edge,
        `checker=${r_checker.edge}, smooth=${r_smooth.edge}`);
expect('checker edge < smooth edge (strictly)',
        r_checker.edge < r_smooth.edge,
        `checker=${r_checker.edge}, smooth=${r_smooth.edge}`);

const triBudget = r_checker.diagnostics.triBudget;
expect('checker estTriangles ≤ triBudget',
        r_checker.diagnostics.estTriangles <= triBudget * 1.001,
        `est=${r_checker.diagnostics.estTriangles} budget=${triBudget}`);
expect('smooth estTriangles ≤ triBudget',
        r_smooth.diagnostics.estTriangles <= triBudget * 1.001,
        `est=${r_smooth.diagnostics.estTriangles} budget=${triBudget}`);

section('computeSmartResolution — laserPlate (planar XY)');
if (plate) {
  const plateBounds = computeBounds(plate);
  const s = { ...baseSettings, mappingMode: MODE_PLANAR_XY, scaleU: 0.5, scaleV: 0.5 };
  const r_p_smooth  = computeSmartResolution({ geometry: plate, bounds: plateBounds, settings: s, texture: tx_smooth });
  const r_p_checker = computeSmartResolution({ geometry: plate, bounds: plateBounds, settings: s, texture: tx_checker });
  console.log(`  smooth  : edge=${r_p_smooth.edge}  estTris=${Math.round(r_p_smooth.diagnostics.estTriangles)}`);
  console.log(`  checker : edge=${r_p_checker.edge}  estTris=${Math.round(r_p_checker.diagnostics.estTriangles)}`);
  expect('plate checker ≤ plate smooth', r_p_checker.edge <= r_p_smooth.edge);
  expect('plate edges within sane bounds [0.05, 5.0]',
          r_p_smooth.edge >= 0.05 && r_p_smooth.edge <= 5.0 &&
          r_p_checker.edge >= 0.05 && r_p_checker.edge <= 5.0);
}

section('computeSmartResolution — budget floor kicks in');
// Force a tiny pix_mm via a small scale so the texture wants a very fine edge,
// then crank maxTriangles down so the budget floor must intervene.
const tinyScaleSettings = {
  ...baseSettings,
  mappingMode: MODE_PLANAR_XY,
  scaleU: 0.05, scaleV: 0.05,  // very fine periodic texture
  maxTriangles: 50_000,
};
// Use the 100mm cube here too — surface area must be large enough for the
// budget floor to exceed the detail edge (small cubes give tiny floors).
const r_budget = computeSmartResolution({
  geometry: cube, bounds: cubeBounds, settings: tinyScaleSettings, texture: tx_checker,
});
console.log(`  budget-clamped : edge=${r_budget.edge}  detailEdge=${r_budget.diagnostics.detailEdge.toFixed(4)}  budgetEdge=${r_budget.diagnostics.budgetEdge.toFixed(4)}  estTris=${Math.round(r_budget.diagnostics.estTriangles)}  triBudget=${r_budget.diagnostics.triBudget}`);
expect('budgetEdge > detailEdge (floor active)',
        r_budget.diagnostics.budgetEdge > r_budget.diagnostics.detailEdge);
expect('budgetClamped flag set', r_budget.diagnostics.budgetClamped === true);
expect('estTriangles ≤ triBudget × 1.05',
        r_budget.diagnostics.estTriangles <= r_budget.diagnostics.triBudget * 1.05,
        `est=${r_budget.diagnostics.estTriangles} budget=${r_budget.diagnostics.triBudget}`);

section('computeSmartResolution — degenerate inputs');
const rNull = computeSmartResolution({ geometry: null, bounds: null, settings: baseSettings, texture: null });
expect('null inputs return null', rNull === null);

console.log(_failed === 0
  ? `\nALL TESTS PASSED`
  : `\n${_failed} TEST(S) FAILED`);
process.exit(_failed === 0 ? 0 : 1);
