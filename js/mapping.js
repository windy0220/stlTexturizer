/**
 * CPU-side UV mapping — exact JavaScript mirror of the GLSL in previewMaterial.js.
 * All functions take Three.js Vector3 objects for position/normal and
 * a bounds object { min, max, center, size } (all THREE.Vector3).
 */

export const MODE_PLANAR_XY   = 0;
export const MODE_PLANAR_XZ   = 1;
export const MODE_PLANAR_YZ   = 2;
export const MODE_CYLINDRICAL = 3;
export const MODE_SPHERICAL   = 4;
export const MODE_TRIPLANAR   = 5;
export const MODE_CUBIC       = 6;

const TWO_PI = Math.PI * 2;
const CUBIC_AXIS_EPSILON = 1e-4;

export function getDominantCubicAxis(normal) {
  const ax = Math.abs(normal.x);
  const ay = Math.abs(normal.y);
  const az = Math.abs(normal.z);

  // Treat near-ties as an intentional tie so 45° faces pick one stable axis
  // instead of flipping projection due to tiny normal jitter between triangles.
  if (ax >= ay - CUBIC_AXIS_EPSILON && ax >= az - CUBIC_AXIS_EPSILON) return 'x';
  if (ay >= az - CUBIC_AXIS_EPSILON) return 'y';
  return 'z';
}

/**
 * Compute normalised UV coordinates [0, 1) (tiling) for a vertex.
 *
 * @param {{ x:number, y:number, z:number }} pos      vertex position
 * @param {{ x:number, y:number, z:number }} normal   vertex normal (unit)
 * @param {number}  mode    one of the MODE_* constants
 * @param {{ scaleU:number, scaleV:number, offsetU:number, offsetV:number }} settings
 * @param {{ min, max, center, size }} bounds           THREE.Vector3 fields
 * @returns {{ u:number, v:number }}                    tiled UV after scale+offset
 */
export function computeUV(pos, normal, mode, settings, bounds) {
  const { min, size, center } = bounds;
  const { scaleU, scaleV, offsetU, offsetV } = settings;
  const rotRad = (settings.rotation ?? 0) * Math.PI / 180;
  const maxDim = Math.max(size.x, size.y, size.z);
  const md     = Math.max(maxDim, 1e-6);

  let u = 0, v = 0;

  switch (mode) {

    case MODE_PLANAR_XY: {
      u = (pos.x - min.x) / md;
      v = (pos.y - min.y) / md;
      break;
    }

    case MODE_PLANAR_XZ: {
      u = (pos.x - min.x) / md;
      v = (pos.z - min.z) / md;
      break;
    }

    case MODE_PLANAR_YZ: {
      u = (pos.y - min.y) / md;
      v = (pos.z - min.z) / md;
      break;
    }

    case MODE_CYLINDRICAL: {
      // mappingBlend=0 → pure side projection for all faces (original behaviour, no cap seam).
      // mappingBlend>0 → smooth side↔cap blend; zone half-width = blend*0.20.
      const r  = Math.max(size.x, size.y) * 0.5;
      const C  = TWO_PI * Math.max(r, 1e-6);
      const rx = pos.x - center.x;
      const ry = pos.y - center.y;
      const blend = settings.mappingBlend ?? 0.0;
      const theta = Math.atan2(ry, rx);
      const uSide = (theta / TWO_PI) + 0.5;
      const vSide = (pos.z - min.z) / C;
      if (blend <= 0.001) {
        return applyTransform(uSide, vSide, scaleU, scaleV, offsetU, offsetV, rotRad);
      }
      const blendHalf = blend * 0.20;
      const absnz = Math.abs(normal.z);
      const capW = Math.max(0, Math.min(1, (absnz - (0.7 - blendHalf)) / (2 * blendHalf + 1e-6)));
      if (capW <= 0) {
        return applyTransform(uSide, vSide, scaleU, scaleV, offsetU, offsetV, rotRad);
      }
      const uCap  = rx / C + 0.5;
      const vCap  = ry / C + 0.5;
      if (capW >= 1) {
        return applyTransform(uCap, vCap, scaleU, scaleV, offsetU, offsetV, rotRad);
      }
      // Return two separate samples so displacement.js blends the *heights*,
      // not the UV coordinates (blending atan2-based and planar UVs directly
      // produces garbage values in the transition zone).
      const tSide = applyTransform(uSide, vSide, scaleU, scaleV, offsetU, offsetV, rotRad);
      const tCap  = applyTransform(uCap,  vCap,  scaleU, scaleV, offsetU, offsetV, rotRad);
      return {
        triplanar: true,
        samples: [
          { u: tSide.u, v: tSide.v, w: 1 - capW },
          { u: tCap.u,  v: tCap.v,  w: capW },
        ],
      };
    }

    case MODE_SPHERICAL: {
      const rx = pos.x - center.x;
      const ry = pos.y - center.y;
      const rz = pos.z - center.z;
      const r  = Math.sqrt(rx*rx + ry*ry + rz*rz);
      const phi   = Math.acos(Math.max(-1, Math.min(1, rz / Math.max(r, 1e-6)))); // [0, PI], Z is up
      const theta = Math.atan2(ry, rx);              // [-PI, PI]
      u = (theta / TWO_PI) + 0.5;
      v = phi / Math.PI;
      break;
    }

    case MODE_CUBIC: {
      let uRaw, vRaw;
      switch (getDominantCubicAxis(normal)) {
        case 'x':
          uRaw = (pos.y - min.y) / md;
          vRaw = (pos.z - min.z) / md;
          break;
        case 'y':
          uRaw = (pos.x - min.x) / md;
          vRaw = (pos.z - min.z) / md;
          break;
        default:
          uRaw = (pos.x - min.x) / md;
          vRaw = (pos.y - min.y) / md;
          break;
      }
      return applyTransform(uRaw, vRaw, scaleU, scaleV, offsetU, offsetV, rotRad);
    }

    case MODE_TRIPLANAR:
    default: {
      // World-space normal blending
      const ax = Math.abs(normal.x);
      const ay = Math.abs(normal.y);
      const az = Math.abs(normal.z);
      const pw = 4.0;
      const bx = Math.pow(ax, pw);
      const by = Math.pow(ay, pw);
      const bz = Math.pow(az, pw);
      const sum = bx + by + bz + 1e-6;
      const wx = bx / sum;
      const wy = by / sum;
      const wz = bz / sum;

      const uvXY = {
        u: (pos.x - min.x) / md,
        v: (pos.y - min.y) / md,
        w: wz,
      };
      const uvXZ = {
        u: (pos.x - min.x) / md,
        v: (pos.z - min.z) / md,
        w: wy,
      };
      const uvYZ = {
        u: (pos.y - min.y) / md,
        v: (pos.z - min.z) / md,
        w: wx,
      };

      // Apply scale+offset+rotation and tile each independently
      return {
        triplanar: true,
        samples: [
          { ...applyTransform(uvXY.u, uvXY.v, scaleU, scaleV, offsetU, offsetV, rotRad), w: uvXY.w },
          { ...applyTransform(uvXZ.u, uvXZ.v, scaleU, scaleV, offsetU, offsetV, rotRad), w: uvXZ.w },
          { ...applyTransform(uvYZ.u, uvYZ.v, scaleU, scaleV, offsetU, offsetV, rotRad), w: uvYZ.w },
        ],
      };
    }
  }

  return applyTransform(u, v, scaleU, scaleV, offsetU, offsetV, rotRad);
}

function applyTransform(u, v, scaleU, scaleV, offsetU, offsetV, rotRad) {
  let uu = u / scaleU + offsetU;
  let vv = v / scaleV + offsetV;
  if (rotRad !== 0) {
    const c = Math.cos(rotRad), s = Math.sin(rotRad);
    uu -= 0.5; vv -= 0.5;
    const ru = c * uu - s * vv;
    const rv = s * uu + c * vv;
    uu = ru + 0.5; vv = rv + 0.5;
  }
  return { triplanar: false, u: fract(uu), v: fract(vv) };
}

/** Fractional part, always positive (mirrors GLSL fract) */
function fract(x) { return x - Math.floor(x); }
