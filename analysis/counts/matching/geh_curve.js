// This code runs inside the shared map-controls closure, using its deck instance.
const gehPlot = document.getElementById("counts-geh-plot");
const gehThreshold = document.getElementById("counts-geh-threshold");
const gehReadout = document.getElementById("counts-geh-readout");
let gehValues = [];
let gehMaximum = 10;
const gehBox = {left: 48, right: 328, top: 24, bottom: 168};

function gehCountBelow(threshold) {
  let low = 0, high = gehValues.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (gehValues[middle] <= threshold) low = middle + 1;
    else high = middle;
  }
  return low;
}

function gehSvg(tag, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  gehPlot.appendChild(element);
  return element;
}

function showGehThreshold() {
  if (!gehPlot || !document.createElementNS) return;
  const threshold = Number(gehThreshold.value);
  const valid = gehThreshold.value.trim() !== "" && Number.isFinite(threshold) && threshold >= 0;
  const count = valid ? gehCountBelow(threshold) : 0;
  gehReadout.textContent = !valid ? "Enter a non-negative GEH threshold."
    : !gehValues.length ? "No visible points with finite GEH."
    : `GEH ≤ ${threshold}: ${count.toLocaleString()} / ${gehValues.length.toLocaleString()} points (${(100 * count / gehValues.length).toFixed(2)}%)`;
  gehPlot.querySelectorAll(".geh-cursor").forEach(element => element.remove());
  if (!valid || !gehValues.length || threshold > gehMaximum) return;
  const x = gehBox.left + threshold / gehMaximum * (gehBox.right - gehBox.left);
  const y = gehBox.bottom - count / gehValues.length * (gehBox.bottom - gehBox.top);
  gehSvg("line", {class: "geh-cursor", x1: x, x2: x, y1: gehBox.top, y2: gehBox.bottom,
                  stroke: "#d95f02", "stroke-dasharray": "3 3"});
  gehSvg("circle", {class: "geh-cursor", cx: x, cy: y, r: 3, fill: "#d95f02"});
}

function updateGehCurve() {
  if (!gehPlot || !document.createElementNS || !deck || !deck.props) return;
  // Count each station once: accuracy-center and network layers are excluded.
  gehValues = deck.props.layers.filter(layer =>
    layer.id.startsWith("counts-points-") && layer.props.visible !== false
  ).flatMap(layer => layer.props.data).map(point => point.geh)
    .filter(value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)))
    .map(Number).filter(value => value >= 0).sort((a, b) => a - b);
  gehMaximum = Math.max(10, Math.ceil(gehValues[gehValues.length - 1] || 0));
  gehPlot.replaceChildren();
  const x = value => gehBox.left + value / gehMaximum * (gehBox.right - gehBox.left);
  const y = count => gehBox.bottom - count / (gehValues.length || 1) * (gehBox.bottom - gehBox.top);
  for (const count of new Set([0, 1, 2, 3, 4].map(tick => Math.round(gehValues.length * tick / 4)))) {
    gehSvg("line", {x1: gehBox.left, x2: gehBox.right, y1: y(count), y2: y(count), stroke: "#ddd"});
    gehSvg("text", {x: gehBox.left - 5, y: y(count) + 3, "text-anchor": "end", "font-size": 10}, count);
  }
  for (let tick = 0; tick <= 4; tick++) {
    const value = gehMaximum * tick / 4;
    gehSvg("text", {x: x(value), y: gehBox.bottom + 16, "text-anchor": "middle", "font-size": 10}, Number(value.toFixed(1)));
  }
  gehSvg("text", {x: gehBox.left, y: 13, "font-size": 11}, `Cumulative points (n=${gehValues.length})`);
  gehSvg("text", {x: (gehBox.left + gehBox.right) / 2, y: 202, "text-anchor": "middle", "font-size": 11}, "GEH threshold");
  let path = `M${x(0)},${y(0)}`;
  for (let index = 0; index < gehValues.length; index++) {
    path += `H${x(gehValues[index])}V${y(index + 1)}`;
  }
  path += `H${x(gehMaximum)}`;
  gehSvg("path", {d: path, fill: "none", stroke: "#2166ac", "stroke-width": 2});
  showGehThreshold();
}

if (gehPlot) {
  gehPlot.addEventListener("pointermove", event => {
    const rectangle = gehPlot.getBoundingClientRect();
    const x = (event.clientX - rectangle.left) / rectangle.width * 340;
    const value = Math.max(0, Math.min(gehMaximum,
      (x - gehBox.left) / (gehBox.right - gehBox.left) * gehMaximum));
    gehThreshold.value = value.toFixed(2);
    showGehThreshold();
  });
  gehThreshold.addEventListener("input", showGehThreshold);
}
