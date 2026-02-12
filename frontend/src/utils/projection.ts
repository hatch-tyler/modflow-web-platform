import proj4 from 'proj4'

// Common MODFLOW projection definitions
const PROJ_DEFS: Record<number, string> = {
  // UTM zones (North) — explicit defs for common zones
  32601: '+proj=utm +zone=1 +datum=WGS84 +units=m +no_defs',
  32610: '+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs',
  32611: '+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs',
  32612: '+proj=utm +zone=12 +datum=WGS84 +units=m +no_defs',
  32613: '+proj=utm +zone=13 +datum=WGS84 +units=m +no_defs',
  32614: '+proj=utm +zone=14 +datum=WGS84 +units=m +no_defs',
  32615: '+proj=utm +zone=15 +datum=WGS84 +units=m +no_defs',
  32616: '+proj=utm +zone=16 +datum=WGS84 +units=m +no_defs',
  32617: '+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs',
  32618: '+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs',
  32619: '+proj=utm +zone=19 +datum=WGS84 +units=m +no_defs',
  // State Plane NAD83 (common)
  2227: '+proj=lcc +lat_1=38.43333333333333 +lat_2=37.06666666666667 +lat_0=36.5 +lon_0=-120.5 +x_0=2000000 +y_0=500000 +datum=NAD83 +units=ft +no_defs',
  2229: '+proj=lcc +lat_1=35.46666666666667 +lat_2=34.03333333333333 +lat_0=33.5 +lon_0=-118 +x_0=2000000 +y_0=500000 +datum=NAD83 +units=ft +no_defs',
}

/** Build a proj4 converter from the given EPSG to WGS84 (EPSG:4326). Returns null on failure. */
export function getProjection(epsg: number): proj4.Converter | null {
  try {
    // Registered explicit defs
    if (PROJ_DEFS[epsg]) {
      return proj4(PROJ_DEFS[epsg], 'EPSG:4326')
    }
    // UTM North pattern
    if (epsg >= 32601 && epsg <= 32660) {
      const zone = epsg - 32600
      return proj4(
        `+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs`,
        'EPSG:4326',
      )
    }
    // UTM South pattern
    if (epsg >= 32701 && epsg <= 32760) {
      const zone = epsg - 32700
      return proj4(
        `+proj=utm +zone=${zone} +south +datum=WGS84 +units=m +no_defs`,
        'EPSG:4326',
      )
    }
    // Fallback to proj4 built-in EPSG database
    return proj4(`EPSG:${epsg}`, 'EPSG:4326')
  } catch {
    return null
  }
}
