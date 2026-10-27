'use strict';
// Rendering and input layer only: all computation happens in Python.
// Geometry is fetched once from app://data; afterwards only the per-part
// neighborhood array (Uint16) is re-fetched when assignments change.

const {Deck, MapView, SolidPolygonLayer, PathLayer, ScatterplotLayer, WebMercatorViewport, FlyToInterpolator} = deck;

const OUTLINE_MIN_ZOOM = 15;
const ALPHA_SELECTED = 220;
const ALPHA_OTHER = 110;

const state = {
  bridge: null,
  deck: null,
  starts: null,       // Uint32Array, nParts + 1
  fillData: null,     // binary data object for the fill layer (identity kept stable)
  outlineData: null,
  positions: null,    // Float64Array of all vertices (kept for highlight outlines)
  salesData: null,    // binary data for sale points, or null
  highlight: [],      // [{path}] outlines of the highlighted parcel
  partNbhd: null,     // Uint16Array, neighborhood index per part
  palette: [],        // [{name, parent, color}]
  selected: 0,
  colorVersion: 0,
  zoom: 0,
  pendingTimer: null, // {name, t0}: reported after the next render
};

const statusEl = document.getElementById('status');
const setStatus = (text) => { statusEl.textContent = text; };

async function fetchArray(name, Type) {
  const response = await fetch(`app://data/${name}`);
  if (!response.ok) throw new Error(`fetch ${name}: ${response.status}`);
  return new Type(await response.arrayBuffer());
}

function buildColors() {
  const {starts, partNbhd, palette, selected} = state;
  const lut = new Uint8Array(palette.length * 4);
  palette.forEach((n, i) => {
    lut.set(n.color, i * 4);
    lut[i * 4 + 3] = i === selected ? ALPHA_SELECTED : ALPHA_OTHER;
  });
  const nParts = partNbhd.length;
  const colors = new Uint8Array(starts[nParts] * 4);
  for (let p = 0; p < nParts; p++) {
    const o = partNbhd[p] * 4;
    const r = lut[o], g = lut[o + 1], b = lut[o + 2], a = lut[o + 3];
    for (let v = starts[p] * 4, end = starts[p + 1] * 4; v < end; v += 4) {
      colors[v] = r; colors[v + 1] = g; colors[v + 2] = b; colors[v + 3] = a;
    }
  }
  return colors;
}

function recolor() {
  // Swap only the color buffer; the data object's identity is unchanged so
  // deck.gl does not re-tessellate the polygons.
  state.fillData.attributes.getFillColor = {value: buildColors(), size: 4, normalized: true};
  state.colorVersion++;
  render();
}

function render() {
  const layers = [
    new SolidPolygonLayer({
      id: 'parcels',
      data: state.fillData,
      _normalize: false,
      _windingOrder: 'CCW',
      pickable: true,
      updateTriggers: {getFillColor: state.colorVersion},
    }),
    new PathLayer({
      id: 'outlines',
      data: state.outlineData,
      _pathType: 'open',
      getColor: [20, 20, 20, 160],
      getWidth: 1,
      widthUnits: 'pixels',
      visible: state.zoom >= OUTLINE_MIN_ZOOM,
    }),
    new PathLayer({
      id: 'highlight',
      data: state.highlight,
      positionFormat: 'XY',
      getPath: (d) => d.path,
      getColor: [255, 220, 60, 255],
      getWidth: 3,
      widthUnits: 'pixels',
    }),
  ];
  if (state.salesData) {
    layers.push(new ScatterplotLayer({
      id: 'sales',
      data: state.salesData,
      pickable: true,
      radiusUnits: 'pixels',
      getRadius: 4,
      stroked: true,
      getLineColor: [20, 20, 20, 220],
      lineWidthUnits: 'pixels',
      getLineWidth: 1,
    }));
  }
  state.deck.setProps({layers});
}

function startTimer(name, t0 = performance.now()) {
  state.pendingTimer = {name, t0};
}

function onAfterRender() {
  const timer = state.pendingTimer;
  if (!timer) return;
  state.pendingTimer = null;
  // Wait one more frame so the GPU has actually presented the result.
  requestAnimationFrame(() => {
    state.bridge.reportTiming(timer.name, performance.now() - timer.t0);
    if (timer.name === 'first_render' && state.afterFirstRender) {
      const next = state.afterFirstRender;
      state.afterFirstRender = null;
      next();
    }
  });
}

async function onAssignmentChanged(version, paletteJson) {
  const t0 = performance.now();
  state.palette = JSON.parse(paletteJson);
  state.partNbhd = await fetchArray(`assignment?v=${version}`, Uint16Array);
  if (!state.pendingTimer) startTimer('recolor', t0);
  recolor();
}

async function onSalesChanged(version) {
  const [points, colors] = await Promise.all([
    fetchArray(`sale_points?v=${version}`, Float64Array),
    fetchArray(`sale_colors?v=${version}`, Uint8Array),
  ]);
  const length = points.length / 3;
  state.salesData = length ? {
    length,
    attributes: {
      getPosition: {value: points, size: 3},
      getFillColor: {value: colors, size: 4, normalized: true},
    },
  } : null;
  render();
}

function onHighlightRings(rangesJson) {
  state.highlight = JSON.parse(rangesJson).map(([s, e]) => ({path: state.positions.subarray(s * 2, e * 2)}));
  render();
}

function onFlyTo(longitude, latitude, zoom) {
  state.deck.setProps({initialViewState: {
    longitude, latitude, zoom, pitch: 0, bearing: 0,
    transitionDuration: 700, transitionInterpolator: new FlyToInterpolator(),
  }});
}

function onSelectionChanged(index) {
  state.selected = index;
  startTimer('select');
  recolor();
}

// --- Lasso: shift + drag ----------------------------------------------------

function setupLasso(container) {
  const canvas = document.getElementById('lasso');
  const ctx = canvas.getContext('2d');
  let points = null;
  let mode = 'create';  // shift+drag creates a sub-neighborhood; ctrl+shift+drag reassigns
  const styles = {
    create: {fill: 'rgba(255, 220, 60, 0.15)', stroke: 'rgba(255, 220, 60, 0.9)'},
    reassign: {fill: 'rgba(80, 200, 255, 0.15)', stroke: 'rgba(80, 200, 255, 0.95)'},
  };

  const resize = () => { canvas.width = container.clientWidth; canvas.height = container.clientHeight; };
  window.addEventListener('resize', resize);
  resize();

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points || points.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (const [x, y] of points) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.fillStyle = styles[mode].fill;
    ctx.strokeStyle = styles[mode].stroke;
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  };

  // Capture phase so deck.gl never sees shift-drags (it would pan the map).
  container.addEventListener('pointerdown', (e) => {
    if (!e.shiftKey || e.button !== 0) return;
    e.stopPropagation();
    container.setPointerCapture(e.pointerId);
    mode = e.ctrlKey ? 'reassign' : 'create';
    points = [[e.offsetX, e.offsetY]];
  }, true);
  container.addEventListener('pointermove', (e) => {
    if (!points) return;
    e.stopPropagation();
    points.push([e.offsetX, e.offsetY]);
    draw();
  }, true);
  container.addEventListener('pointerup', (e) => {
    if (!points) return;
    e.stopPropagation();
    const screen = points;
    points = null;
    draw();
    if (screen.length < 3) return;
    const viewport = state.deck.getViewports()[0];
    const lngLat = screen.map((p) => viewport.unproject(p));
    startTimer('lasso');
    state.bridge.lasso(JSON.stringify(lngLat), mode);
  }, true);
}

// --- Startup -----------------------------------------------------------------

async function loadData() {
  const t0 = performance.now();
  const meta = await (await fetch('app://data/meta')).json();
  const [positions, starts, ringStarts, indices, partNbhd] = await Promise.all([
    fetchArray('positions', Float64Array),
    fetchArray('starts', Uint32Array),
    fetchArray('rings', Uint32Array),
    fetchArray('indices', Uint32Array),
    fetchArray('assignment?v=0', Uint16Array),
  ]);
  state.bridge.reportTiming('fetch', performance.now() - t0);

  const nParts = partNbhd.length;
  const startIndices = starts.subarray(0, nParts);
  state.starts = starts;
  state.positions = positions;
  state.partNbhd = partNbhd;
  state.palette = meta.palette;
  // Triangles come precomputed (holes included), so deck.gl skips tessellation.
  state.fillData = {
    length: nParts,
    startIndices,
    attributes: {getPolygon: {value: positions, size: 2}, indices},
  };
  // One path per ring, so hole outlines are not joined to their exterior.
  const nRings = ringStarts.length - 1;
  state.outlineData = {
    length: nRings,
    startIndices: ringStarts.subarray(0, nRings),
    attributes: {getPath: {value: positions, size: 2}},
  };

  const container = document.getElementById('map');
  const [minX, minY, maxX, maxY] = meta.bounds;
  const fitted = new WebMercatorViewport({width: container.clientWidth, height: container.clientHeight})
    .fitBounds([[minX, minY], [maxX, maxY]], {padding: 20});
  state.zoom = fitted.zoom;

  state.deck = new Deck({
    parent: container,
    views: new MapView({repeat: false}),
    initialViewState: {longitude: fitted.longitude, latitude: fitted.latitude, zoom: fitted.zoom},
    controller: {dragRotate: false, touchRotate: false, doubleClickZoom: false, keyboard: false},
    getCursor: ({isHovering}) => (isHovering ? 'pointer' : 'grab'),
    onClick: (info) => {
      if (!info.layer || info.index < 0) return;
      if (info.layer.id === 'sales') state.bridge.clickSale(info.index);
      else if (info.layer.id === 'parcels') state.bridge.clickPart(info.index);
    },
    onViewStateChange: ({viewState}) => {
      const crossed = (viewState.zoom >= OUTLINE_MIN_ZOOM) !== (state.zoom >= OUTLINE_MIN_ZOOM);
      state.zoom = viewState.zoom;
      if (crossed) render();
    },
    onAfterRender,
    onError: (error) => state.bridge.log(`deck error: ${error}`),
  });
  setupLasso(container);

  startTimer('first_render', t0);
  recolor();
  state.afterFirstRender = () => onSalesChanged(0);  // sale points once parcels are on screen
  setStatus(`${meta.n_parcels.toLocaleString()} parcels · click: open neighborhood · `
            + `shift+drag: new sub-neighborhood · ctrl+shift+drag: move parcels into the selected neighborhood`);
}

window.addEventListener('error', (e) => state.bridge && state.bridge.log(`js error: ${e.message}`));

new QWebChannel(qt.webChannelTransport, (channel) => {
  state.bridge = channel.objects.bridge;
  state.bridge.assignmentChanged.connect(onAssignmentChanged);
  state.bridge.selectionChanged.connect(onSelectionChanged);
  state.bridge.salesChanged.connect((v) => { if (state.deck) onSalesChanged(v); });
  state.bridge.highlightRings.connect(onHighlightRings);
  state.bridge.flyTo.connect(onFlyTo);
  state.bridge.jsReady();
  loadData().catch((error) => {
    setStatus(`Failed to load: ${error}`);
    state.bridge.log(`load failed: ${error && error.stack || error}`);
  });
});
