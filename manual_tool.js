const files = document.querySelector('#files'),
  image = document.querySelector('#image'),
  stage = document.querySelector('#stage'),
  svg = document.querySelector('#overlay'),
  thumbs = document.querySelector('#thumbs'),
  loupe = document.querySelector('#loupe'),
  status = document.querySelector('#status'),
  info = document.querySelector('#info'),
  correction = document.querySelector('#correction'),
  trimDarkEdges = document.querySelector('#trimDarkEdges'),
  cmsUrl = document.querySelector('#cmsUrl'),
  cmsConnect = document.querySelector('#cmsConnect'),
  cmsConnection = document.querySelector('#cmsConnection'),
  cmsConnected = document.querySelector('#cmsConnected'),
  cmsApprovalLink = document.querySelector('#cmsApprovalLink'),
  cmsSearch = document.querySelector('#cmsSearch'),
  cmsResults = document.querySelector('#cmsResults'),
  cmsSelected = document.querySelector('#cmsSelected'),
  cmsUpload = document.querySelector('#cmsUpload'),
  cmsUploadStatus = document.querySelector('#cmsUploadStatus'),
  cmsPositionInput = document.querySelector('#cmsPosition'),
  cmsOverwriteControl = document.querySelector('#cmsOverwriteControl'),
  cmsOverwrite = document.querySelector('#cmsOverwrite'),
  workTab = document.querySelector('#workTab'),
  settingsTab = document.querySelector('#settingsTab'),
  workView = document.querySelector('#workView'),
  settingsView = document.querySelector('#settingsView'),
  sourceDirectory = document.querySelector('#sourceDirectory'),
  finalStoreDirectory = document.querySelector('#finalStoreDirectory'),
  saveSettingsButton = document.querySelector('#saveSettings'),
  settingsStatus = document.querySelector('#settingsStatus');
let names = [],
  index = 0,
  natural = [0, 0],
  baseNatural = [0, 0],
  corners = [],
  active = -1,
  edgeActive = -1,
  rotateMode = false,
  rotateStartAngle = 0,
  rotateScreenCenter = null,
  dragging = false,
  lineMode = false,
  lineStart = null,
  moveMode = false,
  moveStart = null,
  moveOrigin = null,
  cursorPoint = null,
  cursorTarget = svg,
  suggestionRequest = 0,
  rotation = 0,
  zoom = 1,
  pendingCorners = null,
  pendingSuggestion = null,
  suggestionEligible = true,
  cmsToken = null,
  cmsTokenExpiresAt = 0,
  selectedCmsBook = null,
  cmsPosition = 1,
  cmsSearchTimer = null,
  localSettings = null;
const NS = 'http://www.w3.org/2000/svg';
const CMS_SESSION_KEY = 'bookcropper.cms.session';
const MAX_CMS_POSITION = 2147483647;
const suggestionCache = new Map();
const scanCacheKeys = new Map();
let prefetchedPreview = null;
let prefetchedPreviewKey = '';

function defaultCorners() {
  const [w, h] = natural, m = Math.min(w, h) * .025;
  return [
    [m, m],
    [w - m, m],
    [w - m, h - m],
    [m, h - m]
  ];
}

function path() {
  return names[index];
}

function setFilenameHash() {
  if (path()) {
    history.replaceState(null, '', '#' + encodeURIComponent(path()));
    syncCmsPositionFromFilename();
  }
}

function filenameFromHash() {
  return decodeURIComponent(location.hash.slice(1));
}

function renderThumbs() {
  thumbs.innerHTML = '';
  names.forEach((name, i) => {
    const thumb = document.createElement('img');
    thumb.className = 'thumb';
    thumb.dataset.index = i;
    thumb.title = name;
    thumb.alt = name;
    thumb.src = '/api/thumbnail?path=' + encodeURIComponent(name);
    thumb.onclick = () => {
      index = i;
      files.selectedIndex = index;
      files.dispatchEvent(new Event('change'));
    };
    thumbs.appendChild(thumb);
  });
  updateActiveThumb();
}

function updateActiveThumb() {
  const items = thumbs.querySelectorAll('.thumb');
  items.forEach((thumb, i) => {
    thumb.classList.toggle('active', i === index);
  });
  const activeThumb = items[index];
  if (activeThumb) {
    requestAnimationFrame(() => activeThumb.scrollIntoView({block: 'nearest', inline: 'nearest'}));
  }
}

function setSidebarView(view) {
  const settingsActive = view === 'settings';
  workView.hidden = settingsActive;
  settingsView.hidden = !settingsActive;
  workTab.classList.toggle('active', !settingsActive);
  settingsTab.classList.toggle('active', settingsActive);
  workTab.setAttribute('aria-selected', String(!settingsActive));
  settingsTab.setAttribute('aria-selected', String(settingsActive));
}

function replaceImageList(nextNames, preferredIndex = 0, requestedName = '') {
  names = Array.isArray(nextNames) ? nextNames : [];
  files.replaceChildren();
  names.forEach(name => files.add(new Option(name, name)));
  renderThumbs();
  if (!names.length) {
    index = 0;
    natural = [0, 0];
    baseNatural = [0, 0];
    corners = [];
    image.removeAttribute('src');
    svg.replaceChildren();
    history.replaceState(null, '', location.pathname + location.search);
    status.textContent = 'No scans remaining';
    info.textContent = '';
    return;
  }
  const requestedIndex = requestedName ? names.indexOf(requestedName) : -1;
  index = requestedIndex >= 0
    ? requestedIndex
    : Math.min(Math.max(0, preferredIndex), names.length - 1);
  files.selectedIndex = index;
  files.dispatchEvent(new Event('change'));
}

async function localJson(endpoint, init) {
  const response = await fetch(endpoint, init);
  let data = {};
  try {
    data = await response.json();
  } catch {
    // A local server error may not include JSON.
  }
  if (!response.ok) throw new Error(data.error || `Local request failed (${response.status})`);
  return data;
}

async function loadLocalSettings() {
  try {
    localSettings = await localJson('/api/settings');
    sourceDirectory.value = localSettings.sourceDirectory || '';
    finalStoreDirectory.value = localSettings.finalStoreDirectory || '';
  } catch (error) {
    settingsStatus.textContent = error.message;
  }
  updateCmsUi();
  updateCmsPositionStatus();
}

async function saveLocalSettings() {
  saveSettingsButton.disabled = true;
  settingsStatus.textContent = 'Saving...';
  try {
    const response = await localJson('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sourceDirectory: sourceDirectory.value.trim(),
        finalStoreDirectory: finalStoreDirectory.value.trim()
      })
    });
    localSettings = response;
    suggestionCache.clear();
    scanCacheKeys.clear();
    prefetchedPreview = null;
    prefetchedPreviewKey = '';
    sourceDirectory.value = response.sourceDirectory;
    finalStoreDirectory.value = response.finalStoreDirectory;
    settingsStatus.textContent = 'Settings saved';
    replaceImageList(response.images, 0);
    setSidebarView('work');
    updateCmsUi();
    updateCmsPositionStatus();
  } catch (error) {
    settingsStatus.textContent = error.message;
  } finally {
    saveSettingsButton.disabled = false;
  }
}

function applySuggestion(suggestion) {
  if (!suggestion || !suggestion.corners || !baseNatural[0] || !suggestionEligible) return;
  const next = Number(suggestion.rotation) || 0;
  pendingCorners = rectangleFrom(suggestion.corners.map(p => rotatePoint(p, baseNatural[0], baseNatural[1], next)));
  rotation = next;
  document.querySelector('#angle').value = rotation.toFixed(1);
  pendingSuggestion = null;
  status.textContent = suggestion.note?.startsWith('pre-cropped') ? 'Already full-frame' : 'Auto crop ready';
  load(true);
}

function suggestionFor(name) {
  if (!suggestionCache.has(name)) {
    const request = fetch('/api/suggestion?path=' + encodeURIComponent(name))
      .then(response => response.json())
      .catch(error => {
        suggestionCache.delete(name);
        throw error;
      });
    suggestionCache.set(name, request);
  }
  return suggestionCache.get(name);
}

function scanImageUrl(name, rotationValue = 0, corrected = correction.checked) {
  if (!scanCacheKeys.has(name)) scanCacheKeys.set(name, Date.now());
  return '/api/image?path=' + encodeURIComponent(name)
    + '&rotate=' + rotationValue
    + '&correct=' + (corrected ? '1' : '0')
    + '&cache=' + scanCacheKeys.get(name);
}

function prefetchNextScan() {
  const nextName = names[index + 1];
  if (!nextName) return;
  suggestionFor(nextName).catch(() => {});
  const previewKey = `${nextName}|${Number(correction.checked)}`;
  if (prefetchedPreviewKey !== previewKey) {
    prefetchedPreviewKey = previewKey;
    prefetchedPreview = new Image();
    prefetchedPreview.src = scanImageUrl(nextName);
  }
}

function requestSuggestion() {
  const request = ++suggestionRequest;
  suggestionEligible = true;
  pendingSuggestion = null;
  status.textContent = 'Calculating crop...';
  suggestionFor(path()).then(suggestion => {
    if (request !== suggestionRequest) return;
    if (!suggestion.corners) {
      status.textContent = suggestion.note || 'No crop suggestion';
      return;
    }
    if (baseNatural[0]) applySuggestion(suggestion);
    else pendingSuggestion = suggestion;
    prefetchNextScan();
  }).catch(() => {
    if (request === suggestionRequest) status.textContent = 'Crop suggestion failed';
  });
}

function render() {
  if (!natural[0]) return;
  svg.setAttribute('viewBox', `0 0 ${natural[0]} ${natural[1]}`);
  svg.innerHTML = '';
  const poly = document.createElementNS(NS, 'polygon');
  poly.setAttribute('points', corners.map(p => p.join(',')).join(' '));
  poly.setAttribute('fill', 'none');
  poly.setAttribute('stroke', '#58a6ff');
  poly.setAttribute('stroke-width', Math.max(3, natural[0] / 500));
  svg.appendChild(poly);
  const handleRadius = Math.max(8, natural[0] / 180);
  corners.forEach((p, i) => {
    const zone = document.createElementNS(NS, 'circle');
    zone.classList.add('rotate-zone');
    zone.dataset.rotate = i;
    zone.setAttribute('cx', p[0]);
    zone.setAttribute('cy', p[1]);
    zone.setAttribute('r', handleRadius + 32);
    zone.setAttribute('fill', 'transparent');
    svg.appendChild(zone);
  });
  [
    [0, 1, 'ns-resize'],
    [1, 2, 'ew-resize'],
    [2, 3, 'ns-resize'],
    [3, 0, 'ew-resize']
  ].forEach(([a, b, cursor], i) => {
    const line = document.createElementNS(NS, 'line');
    line.classList.add('edge-hit');
    line.dataset.edge = i;
    line.setAttribute('x1', corners[a][0]);
    line.setAttribute('y1', corners[a][1]);
    line.setAttribute('x2', corners[b][0]);
    line.setAttribute('y2', corners[b][1]);
    line.setAttribute('stroke', 'transparent');
    line.setAttribute('stroke-width', Math.max(24, natural[0] / 80));
    line.style.cursor = cursor;
    svg.appendChild(line);
  });
  corners.forEach((p, i) => {
    const c = document.createElementNS(NS, 'circle');
    c.classList.add('handle');
    c.dataset.i = i;
    c.setAttribute('cx', p[0]);
    c.setAttribute('cy', p[1]);
    c.setAttribute('r', handleRadius);
    svg.appendChild(c);
  });
  info.textContent = `${natural[0]} x ${natural[1]} px | rotation ${rotation}° | ${index+1} / ${names.length}`;
  drawPreview();
}

function fitStage() {
  if (!natural[0]) return;
  const work = document.querySelector('#work'),
    availableW = Math.max(1, work.clientWidth - 36),
    availableH = Math.max(1, work.clientHeight - 36),
    fitScale = Math.min(availableW / natural[0], availableH / natural[1]),
    displayW = natural[0] * fitScale * zoom,
    displayH = natural[1] * fitScale * zoom;
  image.style.width = displayW + 'px';
  image.style.height = displayH + 'px';
  if (zoom === 1) {
    image.style.maxWidth = '100%';
    image.style.maxHeight = 'calc(100vh - 90px)';
    stage.style.maxWidth = '100%';
    stage.style.maxHeight = '100%';
  } else {
    image.style.maxWidth = 'none';
    image.style.maxHeight = 'none';
    stage.style.maxWidth = 'none';
    stage.style.maxHeight = 'none';
  }
  stage.style.width = displayW + 'px';
  stage.style.height = displayH + 'px';
  document.querySelector('#zoomLabel').textContent = Math.round(zoom * 100) + '%';
  render();
}

function setZoom(value) {
  zoom = Math.max(1, Math.min(4, Math.round(value * 20) / 20));
  document.querySelector('#work').classList.toggle('zoomed', zoom > 1);
  requestAnimationFrame(fitStage);
}

function rotatePoint(p, w, h, angle) {
  const a = angle * Math.PI / 180,
    c = Math.cos(a),
    s = Math.sin(a),
    raw = [
      [0, 0],
      [w, 0],
      [w, h],
      [0, h]
    ].map(q => [c * q[0] + s * q[1], -s * q[0] + c * q[1]]),
    minX = Math.min(...raw.map(q => q[0])),
    minY = Math.min(...raw.map(q => q[1]));
  return [c * p[0] + s * p[1] - minX, -s * p[0] + c * p[1] - minY];
}

function unrotatePoint(p, w, h, angle) {
  const a = angle * Math.PI / 180,
    c = Math.cos(a),
    s = Math.sin(a),
    raw = [
      [0, 0],
      [w, 0],
      [w, h],
      [0, h]
    ].map(q => [c * q[0] + s * q[1], -s * q[0] + c * q[1]]),
    minX = Math.min(...raw.map(q => q[0])),
    minY = Math.min(...raw.map(q => q[1])),
    x = p[0] + minX,
    y = p[1] + minY;
  return [c * x - s * y, s * x + c * y];
}

function rectangleFrom(points) {
  const xs = points.map(p => p[0]),
    ys = points.map(p => p[1]),
    l = Math.min(...xs),
    r = Math.max(...xs),
    t = Math.min(...ys),
    b = Math.max(...ys);
  return [
    [l, t],
    [r, t],
    [r, b],
    [l, b]
  ];
}

function load(keepCorners = false) {
  const corrected = correction.checked ? '1' : '0';
  image.onload = () => {
    image.style.transform = '';
    natural = [image.naturalWidth, image.naturalHeight];
    if (!baseNatural[0] || !keepCorners) baseNatural = rotation === 0 ? [...natural] : baseNatural;
    corners = keepCorners && pendingCorners ? pendingCorners : defaultCorners();
    clampCorners();
    if (pendingSuggestion) applySuggestion(pendingSuggestion);
    pendingCorners = null;
    requestAnimationFrame(fitStage);
  };
  image.onerror = () => {
    status.textContent = 'Unable to load preview';
  };
  image.src = scanImageUrl(path(), rotation, corrected === '1');
}

function setRotation(value) {
  suggestionEligible = false;
  const next = Number(value) || 0;
  if (baseNatural[0] && corners.length) {
    const base = corners.map(p => unrotatePoint(p, baseNatural[0], baseNatural[1], rotation));
    pendingCorners = rectangleFrom(base.map(p => rotatePoint(p, baseNatural[0], baseNatural[1], next)));
  }
  rotation = next;
  document.querySelector('#angle').value = rotation.toFixed(1);
  load(true);
}

function rotateBy(delta) {
  setRotation(rotation + delta);
}

function move(delta) {
  index = Math.max(0, Math.min(names.length - 1, index + delta));
  files.selectedIndex = index;
  setFilenameHash();
  updateActiveThumb();
  rotation = 0;
  document.querySelector('#angle').value = '0.0';
  load();
  requestSuggestion();
}

function canvasPoint(e) {
  const r = svg.getBoundingClientRect();
  return [(e.clientX - r.left) * natural[0] / r.width, (e.clientY - r.top) * natural[1] / r.height];
}

function drawLoupe(p) {
  const ctx = loupe.getContext('2d');
  ctx.clearRect(0, 0, loupe.width, loupe.height);
  if (!document.querySelector('#loupeToggle').checked) return;
  const zoom = Number(document.querySelector('#loupeZoom').value),
    span = loupe.width / zoom;
  const sx = Math.max(0, Math.min(natural[0] - span, p[0] - span / 2)),
    sy = Math.max(0, Math.min(natural[1] - span, p[1] - span / 2));
  ctx.drawImage(image, sx, sy, span, span, 0, 0, loupe.width, loupe.height);
  ctx.strokeStyle = '#ffdc55';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(loupe.width / 2, 0);
  ctx.lineTo(loupe.width / 2, loupe.height);
  ctx.moveTo(0, loupe.height / 2);
  ctx.lineTo(loupe.width, loupe.height / 2);
  ctx.stroke();
}

function clampCorners() {
  corners.forEach(p => {
    p[0] = Math.max(0, Math.min(natural[0], p[0]));
    p[1] = Math.max(0, Math.min(natural[1], p[1]));
  });
}

function pointInsideRectangle(p) {
  return p[0] >= corners[0][0] && p[0] <= corners[1][0] &&
    p[1] >= corners[0][1] && p[1] <= corners[2][1];
}

function updateCursor(point, target, optionHeld) {
  cursorPoint = point;
  cursorTarget = target;
  if (optionHeld && target?.dataset.edge === undefined && pointInsideRectangle(point)) {
    svg.classList.add('option-move');
    svg.classList.remove('option-straighten');
  } else if (optionHeld) {
    svg.classList.add('option-straighten');
    svg.classList.remove('option-move');
  } else {
    svg.classList.remove('option-move', 'option-straighten');
  }
}

svg.addEventListener('pointerdown', e => {
  suggestionEligible = false;
  const p = canvasPoint(e);
  updateCursor(p, e.target, e.altKey);
  if (e.altKey && e.target.dataset.edge === undefined && pointInsideRectangle(p)) {
    moveMode = true;
    moveStart = p;
    moveOrigin = corners.map(corner => [...corner]);
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    return;
  }
  if (e.altKey) {
    lineMode = true;
    lineStart = p;
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    drawStraightenLine(p, p);
    return;
  }
  if (e.target.dataset.rotate !== undefined) {
    const r = stage.getBoundingClientRect();
    rotateMode = true;
    rotateScreenCenter = [r.left + r.width / 2, r.top + r.height / 2];
    rotateStartAngle = Math.atan2(e.clientY - rotateScreenCenter[1], e.clientX - rotateScreenCenter[0]);
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    return;
  }
  if (e.target.dataset.edge !== undefined) {
    edgeActive = +e.target.dataset.edge;
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    return;
  }
  if (e.target.dataset.i === undefined) return;
  active = +e.target.dataset.i;
  dragging = true;
  svg.setPointerCapture(e.pointerId);
});

function drawStraightenLine(a, b) {
  let line = svg.querySelector('.straighten-line');
  if (!line) {
    line = document.createElementNS(NS, 'line');
    line.classList.add('straighten-line');
    line.setAttribute('stroke', '#ffdc55');
    line.setAttribute('stroke-width', Math.max(3, natural[0] / 500));
    line.setAttribute('stroke-dasharray', '12 8');
    svg.appendChild(line);
  }
  line.setAttribute('x1', a[0]);
  line.setAttribute('y1', a[1]);
  line.setAttribute('x2', b[0]);
  line.setAttribute('y2', b[1]);
}
svg.addEventListener('pointermove', e => {
  const p = canvasPoint(e);
  updateCursor(p, e.target, e.altKey);
  drawLoupe(p);
  if (!dragging) return;
  if (lineMode) {
    drawStraightenLine(lineStart, p);
    return;
  }
  if (moveMode) {
    const dx = p[0] - moveStart[0],
      dy = p[1] - moveStart[1];
    corners = moveOrigin.map(corner => [corner[0] + dx, corner[1] + dy]);
    clampCorners();
    render();
    return;
  }
  if (rotateMode) {
    let delta = (Math.atan2(e.clientY - rotateScreenCenter[1], e.clientX - rotateScreenCenter[0]) - rotateStartAngle) * 180 / Math.PI;
    if (delta > 180) delta -= 360;
    if (delta <= -180) delta += 360;
    image.style.transform = `rotate(${delta}deg)`;
    return;
  }
  const min = 2;
  if (edgeActive === 0) {
    const y = Math.min(p[1], corners[2][1] - min);
    corners[0][1] = y;
    corners[1][1] = y;
  }
  if (edgeActive === 1) {
    const x = Math.max(p[0], corners[0][0] + min);
    corners[1][0] = x;
    corners[2][0] = x;
  }
  if (edgeActive === 2) {
    const y = Math.max(p[1], corners[0][1] + min);
    corners[2][1] = y;
    corners[3][1] = y;
  }
  if (edgeActive === 3) {
    const x = Math.min(p[0], corners[1][0] - min);
    corners[0][0] = x;
    corners[3][0] = x;
  }
  if (edgeActive >= 0) {
    clampCorners();
    render();
    return;
  }
  if (active === 0) {
    corners[0] = [Math.min(p[0], corners[2][0] - min), Math.min(p[1], corners[2][1] - min)];
    corners[1][1] = corners[0][1];
    corners[3][0] = corners[0][0];
  }
  if (active === 1) {
    corners[1] = [Math.max(p[0], corners[0][0] + min), Math.min(p[1], corners[2][1] - min)];
    corners[0][1] = corners[1][1];
    corners[2][0] = corners[1][0];
  }
  if (active === 2) {
    corners[2] = [Math.max(p[0], corners[0][0] + min), Math.max(p[1], corners[0][1] + min)];
    corners[1][0] = corners[2][0];
    corners[3][1] = corners[2][1];
  }
  if (active === 3) {
    corners[3] = [Math.min(p[0], corners[2][0] - min), Math.max(p[1], corners[0][1] + min)];
    corners[2][1] = corners[3][1];
    corners[0][0] = corners[3][0];
  }
  clampCorners();
  render();
});

function alignPendingToLine(a, b, next, orientation) {
  if (!pendingCorners) return;
  const baseA = unrotatePoint(a, baseNatural[0], baseNatural[1], rotation),
    baseB = unrotatePoint(b, baseNatural[0], baseNatural[1], rotation),
    lineA = rotatePoint(baseA, baseNatural[0], baseNatural[1], next),
    lineB = rotatePoint(baseB, baseNatural[0], baseNatural[1], next),
    lineX = (lineA[0] + lineB[0]) / 2,
    lineY = (lineA[1] + lineB[1]) / 2;
  if (orientation === 'horizontal') {
    const topDistance = Math.abs(lineY - pendingCorners[0][1]),
      bottomDistance = Math.abs(lineY - pendingCorners[2][1]);
    if (topDistance <= bottomDistance) {
      const y = Math.min(lineY, pendingCorners[2][1] - 2);
      pendingCorners[0][1] = y;
      pendingCorners[1][1] = y;
    } else {
      const y = Math.max(lineY, pendingCorners[0][1] + 2);
      pendingCorners[2][1] = y;
      pendingCorners[3][1] = y;
    }
  } else {
    const leftDistance = Math.abs(lineX - pendingCorners[0][0]),
      rightDistance = Math.abs(lineX - pendingCorners[1][0]);
    if (leftDistance <= rightDistance) {
      const x = Math.min(lineX, pendingCorners[1][0] - 2);
      pendingCorners[0][0] = x;
      pendingCorners[3][0] = x;
    } else {
      const x = Math.max(lineX, pendingCorners[0][0] + 2);
      pendingCorners[1][0] = x;
      pendingCorners[2][0] = x;
    }
  }
}

function finishPointer(e) {
  if (lineMode && lineStart) {
    const p = canvasPoint(e),
      dx = p[0] - lineStart[0],
      dy = p[1] - lineStart[1];
    if (Math.hypot(dx, dy) > 10) {
      const angle = Math.atan2(dy, dx) * 180 / Math.PI,
        orientation = Math.abs(dx) >= Math.abs(dy) ? 'horizontal' : 'vertical',
        target = orientation === 'horizontal' ? 0 : (dy >= 0 ? 90 : -90);
      let correction = angle - target;
      if (correction > 90) correction -= 180;
      if (correction <= -90) correction += 180;
      const next = rotation + correction;
      setRotation(next);
      alignPendingToLine(lineStart, p, next, orientation);
    } else render();
  } else if (rotateMode) {
    let correction = (Math.atan2(e.clientY - rotateScreenCenter[1], e.clientX - rotateScreenCenter[0]) - rotateStartAngle) * 180 / Math.PI;
    if (correction > 180) correction -= 360;
    if (correction <= -180) correction += 360;
    if (Math.abs(correction) > 0.01) setRotation(rotation - correction);
    else render();
  }
  lineMode = false;
  lineStart = null;
  moveMode = false;
  moveStart = null;
  moveOrigin = null;
  rotateMode = false;
  rotateScreenCenter = null;
  dragging = false;
  active = -1;
  edgeActive = -1;
}
svg.addEventListener('pointerup', finishPointer);
svg.addEventListener('pointercancel', finishPointer);
document.addEventListener('keydown', e => {
  if (e.key === 'Alt' && cursorPoint) updateCursor(cursorPoint, cursorTarget, true);
});
document.addEventListener('keyup', e => {
  if (e.key === 'Alt') svg.classList.remove('option-move', 'option-straighten');
});

function drawPreview() {
  // The scan overview replaces the old crop preview panel.
}

function cropPayload() {
  return {
    path: path(),
    rotation,
    corners,
    correction: correction.checked,
    trimDarkEdges: trimDarkEdges.checked
  };
}

function cmsApiBase() {
  return cmsUrl.value.trim().replace(/\/+$/, '');
}

function cmsErrorMessage(data, fallback) {
  if (Array.isArray(data?.message)) return data.message.join(', ');
  return data?.message || data?.error || fallback;
}

async function cmsRequest(endpoint, init = {}, authenticated = true) {
  const headers = new Headers(init.headers || {});
  headers.set('x-tenant-id', 'jc');
  if (authenticated && cmsToken) headers.set('Authorization', 'Bearer ' + cmsToken);
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(cmsApiBase() + endpoint, {...init, headers});
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = {message: text};
    }
  }
  if (!response.ok) {
    const message = response.status === 401 && authenticated
      ? 'CMS connection expired. Reconnect and retry; the crop was not archived.'
      : cmsErrorMessage(data, `CMS request failed (${response.status})`);
    const error = new Error(message);
    error.status = response.status;
    if (response.status === 401 && authenticated) {
      clearCmsSession('Connection expired', true);
    }
    throw error;
  }
  return data;
}

function setCmsSession(token, expiresAt) {
  cmsToken = token;
  cmsTokenExpiresAt = Number(expiresAt) || 0;
  sessionStorage.setItem(CMS_SESSION_KEY, JSON.stringify({
    token: cmsToken,
    expiresAt: cmsTokenExpiresAt,
    apiBase: cmsApiBase()
  }));
  updateCmsUi();
}

function clearCmsSession(message = 'Disconnected', preserveSelection = false) {
  cmsToken = null;
  cmsTokenExpiresAt = 0;
  cmsOverwrite.checked = false;
  sessionStorage.removeItem(CMS_SESSION_KEY);
  cmsResults.replaceChildren();
  if (!preserveSelection) {
    selectedCmsBook = null;
    cmsSelected.hidden = true;
  }
  cmsUploadStatus.textContent = '';
  cmsConnection.textContent = message;
  updateCmsUi();
}

function cmsPositionIsValid() {
  return Number.isInteger(cmsPosition) && cmsPosition > 0 && cmsPosition <= MAX_CMS_POSITION;
}

function occupiedCmsImage() {
  if (!selectedCmsBook || !cmsPositionIsValid()) return null;
  return (selectedCmsBook.images || []).find(item => Number(item.position) === cmsPosition) || null;
}

function restoreCmsSession() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(CMS_SESSION_KEY) || 'null');
    if (saved?.token && Number(saved.expiresAt) > Date.now()) {
      if (saved.apiBase) cmsUrl.value = saved.apiBase;
      cmsToken = saved.token;
      cmsTokenExpiresAt = Number(saved.expiresAt);
    } else {
      sessionStorage.removeItem(CMS_SESSION_KEY);
    }
  } catch {
    sessionStorage.removeItem(CMS_SESSION_KEY);
  }
  updateCmsUi();
}

function updateCmsUi() {
  const connected = Boolean(cmsToken && cmsTokenExpiresAt > Date.now());
  if (!connected && cmsToken) {
    cmsToken = null;
    cmsTokenExpiresAt = 0;
    sessionStorage.removeItem(CMS_SESSION_KEY);
  }
  cmsConnected.hidden = !connected;
  cmsUrl.disabled = connected;
  cmsConnect.textContent = connected ? 'Disconnect' : 'Connect';
  if (connected) {
    const minutes = Math.max(1, Math.ceil((cmsTokenExpiresAt - Date.now()) / 60000));
    cmsConnection.textContent = `Connected · ${minutes}m`;
  } else if (cmsConnection.textContent.startsWith('Connected')) {
    cmsConnection.textContent = 'Disconnected';
  }
  const occupied = occupiedCmsImage();
  if (!occupied) cmsOverwrite.checked = false;
  cmsOverwriteControl.hidden = !connected || !selectedCmsBook || !occupied;
  cmsUpload.textContent = occupied && cmsPositionIsValid()
    ? `Replace Image ${cmsPosition}`
    : cmsPositionIsValid() ? `Upload as Image ${cmsPosition}` : 'Upload Current Crop';
  cmsUpload.disabled = !connected || !selectedCmsBook || !cmsPositionIsValid()
    || !localSettings?.finalStoreDirectory || Boolean(occupied && !cmsOverwrite.checked);
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function connectCms() {
  if (cmsToken && cmsTokenExpiresAt > Date.now()) {
    try {
      await cmsRequest('/admin/bookcropper/revoke', {method: 'POST'});
    } catch {
      // The local session is cleared even if the token already expired.
    }
    clearCmsSession();
    return;
  }

  const popup = window.open('about:blank', 'bookcropper-pairing', 'popup,width=620,height=720');
  cmsConnect.disabled = true;
  cmsConnection.textContent = 'Starting...';
  cmsUploadStatus.textContent = '';
  try {
    const pairing = await cmsRequest('/admin/auth/bookcropper/pair/start', {
      method: 'POST',
      body: '{}'
    }, false);
    cmsApprovalLink.href = pairing.approvalUrl;
    cmsApprovalLink.hidden = false;
    cmsConnection.textContent = 'Awaiting approval';
    cmsUploadStatus.textContent = 'Approve code ' + pairing.code;
    if (popup) popup.location.href = pairing.approvalUrl;

    while (Date.now() < Number(pairing.expiresAt)) {
      await wait(Number(pairing.pollIntervalMs) || 1000);
      try {
        const exchange = await cmsRequest('/admin/auth/bookcropper/pair/exchange', {
          method: 'POST',
          body: JSON.stringify({code: pairing.code})
        }, false);
        if (exchange.status === 'approved' && exchange.token) {
          setCmsSession(exchange.token, exchange.expiresAt);
          cmsApprovalLink.hidden = true;
          cmsUploadStatus.textContent = 'CMS connection approved';
          if (popup) popup.close();
          cmsSearch.focus();
          return;
        }
      } catch (error) {
        if (error.status !== 429) throw error;
      }
    }
    throw new Error('Pairing code expired');
  } catch (error) {
    cmsConnection.textContent = 'Connection failed';
    cmsUploadStatus.textContent = error.message;
  } finally {
    cmsConnect.disabled = false;
  }
}

function bookLabel(book) {
  return [book.author, book.title, book.year].filter(Boolean).join(' · ');
}

function renderSelectedBook() {
  if (!selectedCmsBook) {
    cmsSelected.hidden = true;
    updateCmsUi();
    return;
  }
  const positions = (selectedCmsBook.images || [])
    .map(item => Number(item.position))
    .filter(position => Number.isInteger(position) && position > 0)
    .sort((a, b) => a - b)
    .join(', ');
  cmsSelected.textContent = bookLabel(selectedCmsBook) + (positions ? ` · images ${positions}` : ' · no images');
  cmsSelected.hidden = false;
  updateCmsUi();
}

function selectCmsBook(book) {
  selectedCmsBook = book;
  cmsOverwrite.checked = false;
  cmsResults.replaceChildren();
  cmsSearch.value = book.title;
  renderSelectedBook();
  updateCmsPositionStatus();
}

function renderCmsBooks(books) {
  cmsResults.replaceChildren();
  if (!books.length) {
    const empty = document.createElement('div');
    empty.className = 'book-result-empty';
    empty.textContent = 'No books found';
    cmsResults.appendChild(empty);
    return;
  }
  books.forEach(book => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'book-result';
    const title = document.createElement('strong');
    title.textContent = book.title;
    const meta = document.createElement('span');
    meta.textContent = [book.author, book.year, book.publisher].filter(Boolean).join(' · ');
    button.append(title, meta);
    button.onclick = () => selectCmsBook(book);
    cmsResults.appendChild(button);
  });
}

async function searchCmsBooks() {
  const query = cmsSearch.value.trim();
  if (query.length < 2 && !/^\d+$/.test(query)) {
    cmsResults.replaceChildren();
    return;
  }
  cmsResults.textContent = 'Searching...';
  try {
    const response = await cmsRequest('/admin/bookcropper/books?q=' + encodeURIComponent(query));
    renderCmsBooks(response.data || []);
  } catch (error) {
    cmsResults.textContent = error.message;
  }
}

function selectCmsPosition(position) {
  const next = Number(position);
  cmsPosition = Number.isInteger(next) && next > 0 && next <= MAX_CMS_POSITION
    ? next
    : null;
  if (cmsPosition !== null) cmsPositionInput.value = String(cmsPosition);
  cmsOverwrite.checked = false;
  updateCmsUi();
  updateCmsPositionStatus();
}

function updateCmsPositionStatus() {
  if (!selectedCmsBook) return;
  if (!localSettings?.finalStoreDirectory) {
    cmsUploadStatus.textContent = 'Set a final-store directory in Settings before uploading';
    return;
  }
  if (!cmsPositionIsValid()) {
    cmsUploadStatus.textContent = 'Enter a positive image position';
    return;
  }
  if (occupiedCmsImage()) {
    cmsUploadStatus.textContent = cmsOverwrite.checked
      ? `Replacement of image ${cmsPosition} unlocked`
      : `Image ${cmsPosition} already exists. Unlock replacement to overwrite.`;
    return;
  }
  cmsUploadStatus.textContent = `Image position ${cmsPosition} is available`;
}

function syncCmsPositionFromFilename() {
  const match = path()?.match(/_(\d+)\.[^.]+$/i);
  if (match) selectCmsPosition(Number(match[1]));
}

async function currentCropBlob() {
  const response = await fetch('/api/crop', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(cropPayload())
  });
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('Restart the cropper server to enable CMS uploads');
    }
    let message = 'Unable to render crop';
    try {
      message = (await response.json()).error || message;
    } catch {
      // Keep the generic message for non-JSON errors.
    }
    throw new Error(message);
  }
  return response.blob();
}

async function finalizeLocalCrop(blob, sourcePath) {
  return localJson('/api/finalize?path=' + encodeURIComponent(sourcePath), {
    method: 'POST',
    headers: {'Content-Type': 'image/jpeg'},
    body: blob
  });
}

async function uploadCurrentCrop() {
  const occupied = occupiedCmsImage();
  if (!selectedCmsBook || !cmsPositionIsValid() || (occupied && !cmsOverwrite.checked)) return;
  cmsUpload.disabled = true;
  cmsUploadStatus.textContent = 'Rendering crop...';
  try {
    const blob = await currentCropBlob();
    const completedPath = path();
    const originalName = completedPath.split('/').pop() || 'crop.jpg';
    const filename = originalName.replace(/\.[^.]+$/, '') + '.jpg';
    const form = new FormData();
    form.append('image', blob, filename);
    if (occupied && cmsOverwrite.checked) form.append('overwrite', 'true');
    cmsUploadStatus.textContent = `Uploading image ${cmsPosition}...`;
    const response = await cmsRequest(
      `/admin/bookcropper/books/${selectedCmsBook.id}/images/${cmsPosition}`,
      {method: 'POST', body: form}
    );
    selectedCmsBook.images = (selectedCmsBook.images || [])
      .filter(item => Number(item.position) !== cmsPosition)
      .concat([{id: response.data.imageId, position: cmsPosition}]);
    cmsOverwrite.checked = false;
    renderSelectedBook();
    cmsUploadStatus.textContent = 'CMS upload complete. Archiving crop...';
    let finalized;
    try {
      finalized = await finalizeLocalCrop(blob, completedPath);
    } catch (error) {
      throw new Error(`CMS upload succeeded; local finalization failed: ${error.message}`);
    }
    const completedPosition = cmsPosition;
    suggestionCache.delete(completedPath);
    scanCacheKeys.delete(completedPath);
    replaceImageList(finalized.images, index);
    cmsUploadStatus.textContent = `Uploaded ${filename} as image ${completedPosition} and archived locally`;
  } catch (error) {
    cmsUploadStatus.textContent = error.message;
  } finally {
    updateCmsUi();
  }
}

files.onchange = () => {
  index = files.selectedIndex;
  cmsOverwrite.checked = false;
  setFilenameHash();
  updateCmsUi();
  updateCmsPositionStatus();
  updateActiveThumb();
  rotation = 0;
  document.querySelector('#angle').value = '0.0';
  load();
  requestSuggestion();
};
document.querySelector('#prev').onclick = () => move(-1);
document.querySelector('#next').onclick = () => move(1);
document.querySelector('#rotateLeft').onclick = () => rotateBy(-90);
document.querySelector('#rotateRight').onclick = () => rotateBy(90);
document.querySelector('#minusDegree').onclick = () => rotateBy(-0.5);
document.querySelector('#plusDegree').onclick = () => rotateBy(0.5);
document.querySelector('#angle').onchange = e => setRotation(e.target.value);
document.querySelector('#reset').onclick = () => {
  corners = defaultCorners();
  render();
};
document.querySelector('#save').onclick = async () => {
  status.textContent = 'Saving...';
  const r = await fetch('/api/save', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(cropPayload())
  });
  const j = await r.json();
  status.textContent = j.error || 'Saved ' + j.saved;
};
correction.onchange = () => {
  pendingCorners = corners.map(p => [...p]);
  load(true);
};
document.querySelector('#zoomOut').onclick = () => setZoom(zoom - 0.05);
document.querySelector('#zoomIn').onclick = () => setZoom(zoom + 0.05);
document.querySelector('#zoomLabel').onclick = () => setZoom(1);
document.querySelector('#loupeToggle').onchange = () => drawLoupe([natural[0] / 2, natural[1] / 2]);
document.querySelector('#loupeZoom').onchange = () => drawLoupe([natural[0] / 2, natural[1] / 2]);
document.addEventListener('keydown', e => {
  if (e.target.closest('input, select, textarea, button')) return;
  if (e.key === 'ArrowLeft' || e.key === 'a') move(-1);
  if (e.key === 'ArrowRight' || e.key === 'd') move(1);
  if (e.key === 'Enter') document.querySelector('#save').click();
});
window.addEventListener('resize', fitStage);
cmsConnect.onclick = connectCms;
cmsSearch.oninput = () => {
  clearTimeout(cmsSearchTimer);
  cmsSearchTimer = setTimeout(searchCmsBooks, 300);
};
cmsSearch.onkeydown = e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    clearTimeout(cmsSearchTimer);
    searchCmsBooks();
  }
};
cmsPositionInput.oninput = () => {
  selectCmsPosition(cmsPositionInput.value);
};
cmsOverwrite.onchange = () => {
  updateCmsUi();
  updateCmsPositionStatus();
};
cmsUpload.onclick = uploadCurrentCrop;
workTab.onclick = () => setSidebarView('work');
settingsTab.onclick = () => {
  setSidebarView('settings');
  sourceDirectory.focus();
};
saveSettingsButton.onclick = saveLocalSettings;
restoreCmsSession();
loadLocalSettings();
fetch('/api/images').then(r => r.json()).then(j => {
  replaceImageList(j.images, 0, filenameFromHash());
});
