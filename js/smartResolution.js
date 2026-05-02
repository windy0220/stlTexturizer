/**
 * Smart resolution — recommend a subdivision target edge length that
 *
 *   (a) is fine enough to resolve the active texture's detail at its
 *       current world-space scale, and
 *   (b) keeps the estimated post-subdivision triangle count below the
 *       hard 20M cap and a multiple of the user's `maxTriangles` budget.
 *
 * The legacy default (bbox-diagonal / 250) ignores both texture frequency
 * and texture period, so it over-meshes smooth textures and under-meshes
 * fine ones.  This module replaces that heuristic when the user clicks
 * the "Smart" button next to the resolution slider.
 */

import { analyzeTexture } from './textureAnalysis.js';
import { computeSurfaceArea } from './stlLoader.js';
import {
  MODE_PLANAR_XY, MODE_PLANAR_XZ, MODE_PLANAR_YZ,
  MODE_CYLINDRICAL, MODE_SPHERICAL,
  MODE_TRIPLANAR, MODE_CUBIC,
} from './mapping.js';

const HARD_CAP_TRIANGLES = 20_000_000;
const HARD_CAP_HEADROOM  = 0.8;        // leave 20% slack under the OOM guard
// Detail floor allowed up to 6× the user's maxTriangles cap.  Most users leave
// maxTriangles at 750K and rely on decimation to crush the post-subdivision
// mesh down — so the budget should be generous and let fine textures over-
// subdivide.  The 0.8 × 20M absolute cap (16M) still bounds the worst case.
const BUDGET_MULTIPLIER  = 6;
// Equilateral-cover constant: triangles per (edge² × area) for an ideal
// equilateral mesh.  Real subdivision output is ~3× this because:
//   • the initial mesh isn't equilateral (skinny/random triangles split unevenly)
//   • adaptive subdivision adds extras to fix T-junctions at refinement boundaries
//   • edges crossing creases / boundaries trigger 1→3 splits instead of 1→4
// 3.0 is the empirical multiplier the user observes in their typical workflow.
const TRIS_PER_AREA_GEOM   = 4 / Math.sqrt(3); // ≈ 2.309
const SUBDIV_GROWTH_FACTOR = 3.0;
const TRIS_PER_AREA_K      = TRIS_PER_AREA_GEOM * SUBDIV_GROWTH_FACTOR; // ≈ 6.93

/**
 * World-space "period" of the texture along U and V — i.e. how many world
 * millimetres correspond to one full UV repeat.  Mirrors the math in
 * mapping.js (computeUV → applyTransform).
 *
 * Returns { periodU_mm, periodV_mm }.  Undefined directions (rare) fall back
 * to the longest planar period so the min() in `computeSmartResolution` does
 * not pick a degenerate axis.
 */
function computeWorldPeriod(settings, bounds) {
  const { size, center, min } = bounds;
  const aspectU = settings.textureAspectU ?? 1;
  const aspectV = settings.textureAspectV ?? 1;
  // Match mapping.js:106-107: effective scale is (settings.scale / aspect)
  const sU = (settings.scaleU || 1e-6) / aspectU;
  const sV = (settings.scaleV || 1e-6) / aspectV;

  const md = Math.max(size.x, size.y, size.z, 1e-6);
  const planar = md * sU; // planar period (any axis — same `md` is used in mapping.js)
  const planarV = md * sV;

  switch (settings.mappingMode) {
    case MODE_PLANAR_XY:
    case MODE_PLANAR_XZ:
    case MODE_PLANAR_YZ:
      return { periodU_mm: planar, periodV_mm: planarV };

    case MODE_CYLINDRICAL: {
      const rDefault = Math.max(size.x, size.y) * 0.5;
      const r = Math.max(settings.cylinderRadius ?? rDefault, 1e-6);
      const C = 2 * Math.PI * r;
      // U: arc length per UV repeat = C × scaleU
      // V: vSide normalised by C, so V period (along Z) = C × scaleV
      return { periodU_mm: C * sU, periodV_mm: C * sV };
    }

    case MODE_SPHERICAL: {
      const r = Math.max(0.5 * Math.max(size.x, size.y, size.z), 1e-6);
      return { periodU_mm: 2 * Math.PI * r * sU, periodV_mm: Math.PI * r * sV };
    }

    case MODE_TRIPLANAR:
    case MODE_CUBIC:
    default:
      // Three planar projections blended by normal — use planar period.
      return { periodU_mm: planar, periodV_mm: planarV };
  }
}

/**
 * @param {object} args
 * @param {THREE.BufferGeometry} args.geometry      Current working geometry.
 * @param {{ min, max, size, center }} args.bounds  Bounds of `geometry`.
 * @param {object} args.settings                    Current settings object (see main.js).
 * @param {{ imageData: ImageData, width: number, height: number }} args.texture
 *        Active texture entry (presetTextures.js shape — must have `imageData`).
 * @returns {{
 *   edge: number,
 *   diagnostics: {
 *     pixelsPerEdge: number,
 *     meanGrad: number,
 *     sharpFrac: number,
 *     pixMm: number,
 *     surfaceArea: number,
 *     detailEdge: number,
 *     budgetEdge: number,
 *     estTriangles: number,
 *     triBudget: number,
 *     budgetClamped: boolean,
 *     edgeClamped: boolean,
 *   }
 * }}
 */
export function computeSmartResolution({ geometry, bounds, settings, texture }) {
  if (!geometry || !bounds || !texture || !texture.imageData) {
    return null;
  }

  // 1. Texture detail → pixels-per-edge.
  const { meanGrad, sharpFrac, pixelsPerEdge } = analyzeTexture(texture.imageData);

  // 2. World-space pixel size.
  const { periodU_mm, periodV_mm } = computeWorldPeriod(settings, bounds);
  const period_mm = Math.min(periodU_mm, periodV_mm);
  const texW = texture.imageData.width || texture.width || 512;
  const texH = texture.imageData.height || texture.height || 512;
  // Use the smaller pixel size across U/V so we resolve the densest direction.
  const pixUmm = periodU_mm / texW;
  const pixVmm = periodV_mm / texH;
  const pixMm = Math.min(pixUmm, pixVmm);

  // 3. Detail-driven edge length (Nyquist-style).
  const detailEdge = pixMm * pixelsPerEdge;

  // 4. Surface area & triangle-budget floor.
  const surfaceArea = computeSurfaceArea(geometry);
  const triBudget = Math.min(
    HARD_CAP_TRIANGLES * HARD_CAP_HEADROOM,
    BUDGET_MULTIPLIER * Math.max(settings.maxTriangles || 750_000, 10_000),
  );
  const budgetEdge = Math.sqrt((TRIS_PER_AREA_K * surfaceArea) / Math.max(triBudget, 1));

  // 5. Final edge: take the larger (coarser) of detail vs budget so neither
  // constraint is violated.
  let edge = Math.max(detailEdge, budgetEdge);
  const budgetClamped = budgetEdge > detailEdge;

  // Sanity clamp: never below 0.05 mm, never coarser than diag/50 (or the
  // 5 mm slider absolute) — matches the legacy default's spirit.
  const diag = Math.sqrt(bounds.size.x ** 2 + bounds.size.y ** 2 + bounds.size.z ** 2);
  const lo = 0.05;
  const hi = Math.min(5.0, diag / 50);
  const preClamp = edge;
  edge = Math.min(Math.max(edge, lo), Math.max(hi, lo));
  const edgeClamped = edge !== preClamp;

  // Round UP to 2 decimals so the slider value never violates the budget
  // floor (rounding down by even 0.005 mm can push estimated triangles past
  // the 4×maxTriangles cap).
  edge = Math.max(lo, Math.ceil(edge * 100) / 100);

  // Estimated triangle count at the chosen edge length.
  const estTriangles = (TRIS_PER_AREA_K * surfaceArea) / (edge * edge);

  return {
    edge,
    diagnostics: {
      pixelsPerEdge,
      meanGrad,
      sharpFrac,
      pixMm,
      period_mm,
      surfaceArea,
      detailEdge,
      budgetEdge,
      estTriangles,
      triBudget,
      budgetClamped,
      edgeClamped,
    },
  };
}
