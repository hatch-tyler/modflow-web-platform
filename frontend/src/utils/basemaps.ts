// MapLibre raster basemap styles for deck.gl Map component

export interface BasemapOption {
  id: string
  label: string
  style: object
  thumbnail: string // CSS gradient or color for the gallery swatch
}

function rasterStyle(label: string, tiles: string[], attribution: string, tileSize = 256): object {
  return {
    version: 8,
    name: label,
    sources: {
      basemap: {
        type: 'raster',
        tiles,
        tileSize,
        attribution,
      },
    },
    layers: [
      {
        id: 'basemap',
        type: 'raster',
        source: 'basemap',
        minzoom: 0,
        maxzoom: 19,
      },
    ],
  }
}

export const BASEMAPS: BasemapOption[] = [
  {
    id: 'terrain',
    label: 'Terrain',
    style: rasterStyle(
      'OpenTopoMap',
      ['https://tile.opentopomap.org/{z}/{x}/{y}.png'],
      '&copy; <a href="https://opentopomap.org">OpenTopoMap</a> contributors',
    ),
    thumbnail: 'linear-gradient(135deg, #c8d8a0 0%, #a8c878 40%, #d4b896 70%, #e8dcc8 100%)',
  },
  {
    id: 'satellite',
    label: 'Satellite',
    style: rasterStyle(
      'ESRI World Imagery',
      ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      '&copy; Esri, Maxar, Earthstar Geographics',
    ),
    thumbnail: 'linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 40%, #1a4a6a 70%, #0a2a3a 100%)',
  },
  {
    id: 'osm',
    label: 'Street Map',
    style: rasterStyle(
      'OpenStreetMap',
      ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    ),
    thumbnail: 'linear-gradient(135deg, #f2efe9 0%, #d5e8d4 40%, #aad3df 70%, #f2efe9 100%)',
  },
]

export const DEFAULT_BASEMAP = BASEMAPS[0] // Terrain
