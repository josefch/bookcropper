const files = document.querySelector('#files'),
  image = document.querySelector('#image'),
  stage = document.querySelector('#stage'),
  svg = document.querySelector('#overlay'),
  thumbs = document.querySelector('#thumbs'),
  loupe = document.querySelector('#loupe'),
  status = document.querySelector('#status'),
  info = document.querySelector('#info'),
  correction = document.querySelector('#correction');
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
  rotation = 0,
  zoom = 1,
  pendingCorners = null,
  pendingSuggestion = null,
  suggestionEligible = true;
const NS = 'http://www.w3.org/2000/svg';

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
  if (path()) history.replaceState(null, '', '#' + encodeURIComponent(path()));
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

function applySuggestion(suggestion) {
  if (!suggestion || !suggestion.corners || !baseNatural[0] || !suggestionEligible) return;
  const next = Number(suggestion.rotation) || 0;
  pendingCorners = rectangleFrom(suggestion.corners.map(p => rotatePoint(p, baseNatural[0], baseNatural[1], next)));
  rotation = next;
  document.querySelector('#angle').value = rotation.toFixed(1);
  pendingSuggestion = null;
  load(true);
}

function requestSuggestion() {
  suggestionEligible = true;
  pendingSuggestion = null;
  fetch('/api/suggestion?path=' + encodeURIComponent(path())).then(r => r.json()).then(suggestion => {
    if (baseNatural[0]) applySuggestion(suggestion);
    else pendingSuggestion = suggestion;
  }).catch(() => {});
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
  image.src = '/api/image?path=' + encodeURIComponent(path()) + '&rotate=' + rotation + '&correct=' + corrected + '&cache=' + Date.now();
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
files.onchange = () => {
  index = files.selectedIndex;
  setFilenameHash();
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
    body: JSON.stringify({
      path: path(),
      rotation,
      corners,
      correction: correction.checked
    })
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
  if (e.key === 'ArrowLeft' || e.key === 'a') move(-1);
  if (e.key === 'ArrowRight' || e.key === 'd') move(1);
  if (e.key === 'Enter') document.querySelector('#save').click();
});
window.addEventListener('resize', fitStage);
fetch('/api/images').then(r => r.json()).then(j => {
  names = j.images;
  names.forEach(n => files.add(new Option(n, n)));
  if (names.length) {
    renderThumbs();
    const requested = filenameFromHash(),
      found = names.indexOf(requested);
    index = found >= 0 ? found : 0;
    files.selectedIndex = index;
    setFilenameHash();
    load();
    requestSuggestion();
  }
});
