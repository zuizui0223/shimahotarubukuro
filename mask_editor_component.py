# -*- coding: utf-8 -*-
"""Interactive image editors used by the Streamlit review app."""
from __future__ import annotations

import base64
import math

import cv2
import numpy as np
import streamlit as st


EDITOR_HTML = """
<div class="mask-editor-shell">
  <canvas title="Drag a vertex, edge handle, line endpoint, or paint stroke"></canvas>
</div>
"""

EDITOR_CSS = """
.mask-editor-shell {
  width: 100%;
  max-width: 720px;
  line-height: 0;
  user-select: none;
  touch-action: none;
}
.mask-editor-shell canvas {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 22%, transparent);
  border-radius: 4px;
  background: white;
  touch-action: none;
}
"""

EDITOR_JS = r"""
export default function ({ parentElement, data, setStateValue }) {
  const canvas = parentElement.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const width = Math.max(1, Math.round(Number(data?.width ?? 720)));
  const height = Math.max(1, Math.round(Number(data?.height ?? 480)));
  const mode = data?.mode ?? "polygon";
  const brush = Math.max(1, Number(data?.brush ?? 20));
  const effect = data?.effect ?? "subtract";

  const cleanPoint = (point) => [
    Math.min(width - 1, Math.max(0, Number(point?.[0] ?? 0))),
    Math.min(height - 1, Math.max(0, Number(point?.[1] ?? 0))),
  ];
  const cleanPolygons = (value) => (Array.isArray(value) ? value : [])
    .map((polygon) => (Array.isArray(polygon) ? polygon.map(cleanPoint) : []))
    .filter((polygon) => polygon.length >= 3);
  const cleanLine = (value) => {
    const points = Array.isArray(value) ? value.slice(0, 2).map(cleanPoint) : [];
    return points.length === 2 ? points : [[width * 0.35, height * 0.5], [width * 0.65, height * 0.5]];
  };

  canvas.width = width;
  canvas.height = height;
  let polygons = cleanPolygons(data?.polygons);
  let line = cleanLine(data?.line);
  let stroke = (Array.isArray(data?.stroke) ? data.stroke : []).map(cleanPoint);
  let drag = null;
  const image = new Image();

  function pointerPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return cleanPoint([
      (event.clientX - rect.left) * width / Math.max(rect.width, 1),
      (event.clientY - rect.top) * height / Math.max(rect.height, 1),
    ]);
  }

  function distanceSquared(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  function segmentProjection(point, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const denom = dx * dx + dy * dy;
    const t = denom > 0 ? Math.min(1, Math.max(0,
      ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom)) : 0;
    const projected = [a[0] + t * dx, a[1] + t * dy];
    return { point: projected, distance2: distanceSquared(point, projected) };
  }

  function nearestVertex(point, radius = 11) {
    let best = null;
    polygons.forEach((polygon, polygonIndex) => {
      polygon.forEach((vertex, vertexIndex) => {
        const distance2 = distanceSquared(point, vertex);
        if (distance2 <= radius * radius && (!best || distance2 < best.distance2)) {
          best = { polygonIndex, vertexIndex, distance2 };
        }
      });
    });
    return best;
  }

  function nearestMidpoint(point, radius = 8) {
    let best = null;
    polygons.forEach((polygon, polygonIndex) => {
      polygon.forEach((a, edgeIndex) => {
        const nextIndex = (edgeIndex + 1) % polygon.length;
        const b = polygon[nextIndex];
        const midpoint = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const distance2 = distanceSquared(point, midpoint);
        if (distance2 <= radius * radius && (!best || distance2 < best.distance2)) {
          best = { polygonIndex, edgeIndex, nextIndex, distance2 };
        }
      });
    });
    return best;
  }

  function nearestSegment(point, radius = 9) {
    let best = null;
    polygons.forEach((polygon, polygonIndex) => {
      polygon.forEach((a, edgeIndex) => {
        const nextIndex = (edgeIndex + 1) % polygon.length;
        const projected = segmentProjection(point, a, polygon[nextIndex]);
        if (projected.distance2 <= radius * radius && (!best || projected.distance2 < best.distance2)) {
          best = { polygonIndex, edgeIndex, nextIndex, ...projected };
        }
      });
    });
    return best;
  }

  function drawLine(points, colour, lineWidth, handles) {
    if (!Array.isArray(points) || points.length !== 2) return;
    ctx.save();
    ctx.strokeStyle = colour;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    ctx.lineTo(points[1][0], points[1][1]);
    ctx.stroke();
    if (handles) {
      points.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
        ctx.fillStyle = index === 0 ? "#ffffff" : "#ffe06a";
        ctx.fill();
        ctx.strokeStyle = colour;
        ctx.lineWidth = 3;
        ctx.stroke();
      });
    }
    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    if (image.complete && image.naturalWidth) {
      ctx.drawImage(image, 0, 0, width, height);
    }

    const contextLines = Array.isArray(data?.context_lines) ? data.context_lines : [];
    contextLines.forEach((item) => drawLine(item.points, item.colour ?? "rgba(70,70,70,.65)", 2, false));

    polygons.forEach((polygon) => {
      ctx.save();
      ctx.beginPath();
      polygon.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point[0], point[1]);
        else ctx.lineTo(point[0], point[1]);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 190, 70, 0.22)";
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 120, 45, 0.98)";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();

      if (mode === "polygon") {
        polygon.forEach((point, index) => {
          const next = polygon[(index + 1) % polygon.length];
          const midpoint = [(point[0] + next[0]) / 2, (point[1] + next[1]) / 2];
          ctx.save();
          ctx.translate(midpoint[0], midpoint[1]);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = "rgba(255,255,255,.95)";
          ctx.strokeStyle = "rgba(0,95,45,.95)";
          ctx.lineWidth = 1.5;
          ctx.fillRect(-3.5, -3.5, 7, 7);
          ctx.strokeRect(-3.5, -3.5, 7, 7);
          ctx.restore();
        });
        polygon.forEach((point) => {
          ctx.beginPath();
          ctx.arc(point[0], point[1], 5, 0, Math.PI * 2);
          ctx.fillStyle = "white";
          ctx.fill();
          ctx.strokeStyle = "rgba(0,95,45,.98)";
          ctx.lineWidth = 2;
          ctx.stroke();
        });
      }
    });

    if (mode === "line") {
      drawLine(line, data?.line_colour ?? "#d92d20", 3, true);
    }
    if (mode === "paint" && stroke.length) {
      ctx.save();
      ctx.strokeStyle = effect === "subtract" ? "rgba(225,35,35,.65)" : "rgba(0,155,210,.65)";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = brush;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(stroke[0][0], stroke[0][1]);
      stroke.slice(1).forEach((point) => ctx.lineTo(point[0], point[1]));
      if (stroke.length === 1) {
        ctx.arc(stroke[0][0], stroke[0][1], brush / 2, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  function onPointerDown(event) {
    event.preventDefault();
    const point = pointerPoint(event);
    try { canvas.setPointerCapture(event.pointerId); } catch (_) {}

    if (mode === "polygon") {
      const vertex = nearestVertex(point);
      if (vertex) {
        drag = { type: "vertex", ...vertex };
      } else {
        const midpoint = nearestMidpoint(point);
        if (midpoint) {
          const polygon = polygons[midpoint.polygonIndex];
          drag = {
            type: "edge",
            ...midpoint,
            start: point,
            startA: [...polygon[midpoint.edgeIndex]],
            startB: [...polygon[midpoint.nextIndex]],
          };
        } else {
          const segment = nearestSegment(point);
          if (!segment) return;
          const insertIndex = segment.edgeIndex + 1;
          polygons[segment.polygonIndex].splice(insertIndex, 0, segment.point);
          drag = { type: "vertex", polygonIndex: segment.polygonIndex, vertexIndex: insertIndex };
        }
      }
    } else if (mode === "line") {
      const d0 = distanceSquared(point, line[0]);
      const d1 = distanceSquared(point, line[1]);
      if (Math.min(d0, d1) <= 13 * 13) {
        drag = { type: "line-end", index: d0 <= d1 ? 0 : 1 };
      } else if (segmentProjection(point, line[0], line[1]).distance2 <= 10 * 10) {
        drag = { type: "line-all", start: point, startLine: line.map((p) => [...p]) };
      } else {
        const index = d0 <= d1 ? 0 : 1;
        line[index] = point;
        drag = { type: "line-end", index };
      }
    } else if (mode === "paint") {
      stroke = [point];
      drag = { type: "stroke" };
    }
    canvas.style.cursor = "grabbing";
    draw();
  }

  function onPointerMove(event) {
    const point = pointerPoint(event);
    if (!drag) {
      if (mode === "polygon") {
        canvas.style.cursor = nearestVertex(point) || nearestMidpoint(point) || nearestSegment(point) ? "grab" : "crosshair";
      } else if (mode === "line") {
        const close = Math.min(distanceSquared(point, line[0]), distanceSquared(point, line[1])) <= 13 * 13;
        canvas.style.cursor = close || segmentProjection(point, line[0], line[1]).distance2 <= 10 * 10 ? "grab" : "crosshair";
      }
      return;
    }
    event.preventDefault();

    if (drag.type === "vertex") {
      polygons[drag.polygonIndex][drag.vertexIndex] = point;
    } else if (drag.type === "edge") {
      const dx = point[0] - drag.start[0];
      const dy = point[1] - drag.start[1];
      const polygon = polygons[drag.polygonIndex];
      polygon[drag.edgeIndex] = cleanPoint([drag.startA[0] + dx, drag.startA[1] + dy]);
      polygon[drag.nextIndex] = cleanPoint([drag.startB[0] + dx, drag.startB[1] + dy]);
    } else if (drag.type === "line-end") {
      line[drag.index] = point;
    } else if (drag.type === "line-all") {
      const dx = point[0] - drag.start[0];
      const dy = point[1] - drag.start[1];
      line = drag.startLine.map((p) => cleanPoint([p[0] + dx, p[1] + dy]));
    } else if (drag.type === "stroke") {
      const previous = stroke[stroke.length - 1];
      if (distanceSquared(previous, point) >= 2.25) stroke.push(point);
    }
    draw();
  }

  function finishPointer(event) {
    if (!drag) return;
    event.preventDefault();
    if (drag.type === "vertex" || drag.type === "edge") setStateValue("polygons", polygons);
    if (drag.type === "line-end" || drag.type === "line-all") setStateValue("line", line);
    if (drag.type === "stroke") setStateValue("stroke", stroke);
    drag = null;
    canvas.style.cursor = "crosshair";
    draw();
  }

  function onDoubleClick(event) {
    if (mode !== "polygon") return;
    const vertex = nearestVertex(pointerPoint(event), 10);
    if (!vertex || polygons[vertex.polygonIndex].length <= 3) return;
    polygons[vertex.polygonIndex].splice(vertex.vertexIndex, 1);
    setStateValue("polygons", polygons);
    draw();
  }

  image.onload = draw;
  image.src = data?.image_url ?? "";
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", finishPointer);
  canvas.addEventListener("pointercancel", finishPointer);
  canvas.addEventListener("dblclick", onDoubleClick);
  draw();

  return () => {
    image.onload = null;
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", finishPointer);
    canvas.removeEventListener("pointercancel", finishPointer);
    canvas.removeEventListener("dblclick", onDoubleClick);
  };
}
"""


image_editor = st.components.v2.component(
    name="corolla_image_editor_v1",
    html=EDITOR_HTML,
    css=EDITOR_CSS,
    js=EDITOR_JS,
)


def component_value(value, name, default):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def bgr_to_jpeg_data_url(image: np.ndarray, width: int, height: int) -> str:
    resized = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("Could not encode editor background")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def mask_to_display_polygons(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    max_vertices: int = 180,
) -> list[list[list[float]]]:
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    display = cv2.resize(crop, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 20:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.75, 0.0012 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx) > max_vertices:
            epsilon *= 1.25
            approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [[float(x), float(y)] for [[x, y]] in approx]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def display_polygons_to_raw(
    polygons,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    image_shape: tuple[int, ...],
) -> list[list[list[float]]]:
    x0, y0, x1, y1 = box
    sx = (x1 - x0) / max(float(width), 1.0)
    sy = (y1 - y0) / max(float(height), 1.0)
    raw_polygons: list[list[list[float]]] = []
    for polygon in polygons or []:
        raw = []
        for point in polygon or []:
            if len(point) < 2 or not all(math.isfinite(float(value)) for value in point[:2]):
                continue
            x = min(max(x0 + float(point[0]) * sx, 0.0), float(image_shape[1] - 1))
            y = min(max(y0 + float(point[1]) * sy, 0.0), float(image_shape[0] - 1))
            raw.append([x, y])
        if len(raw) >= 3:
            raw_polygons.append(raw)
    return raw_polygons


def raw_line_to_display(line, box, width: int, height: int) -> list[list[float]]:
    x0, y0, x1, y1 = box
    sx = float(width) / max(float(x1 - x0), 1.0)
    sy = float(height) / max(float(y1 - y0), 1.0)
    return [[(float(x) - x0) * sx, (float(y) - y0) * sy] for x, y in line]


def display_line_to_raw(line, box, width: int, height: int, image_shape) -> list[list[float]]:
    if not line or len(line) != 2:
        return []
    polygons = display_polygons_to_raw([line + [line[-1]]], box, width, height, image_shape)
    return polygons[0][:2] if polygons else []


def stroke_to_raw_polygons(stroke, brush_px: float, box, width: int, height: int) -> list[list[list[float]]]:
    if not stroke:
        return []
    paint = np.zeros((int(height), int(width)), dtype=np.uint8)
    points = np.array(
        [[round(float(point[0])), round(float(point[1]))] for point in stroke],
        dtype=np.int32,
    )
    thickness = max(1, int(round(float(brush_px))))
    if len(points) == 1:
        cv2.circle(paint, tuple(points[0]), max(1, thickness // 2), 1, -1)
    else:
        cv2.polylines(paint, [points], False, 1, thickness, cv2.LINE_AA)
        cv2.circle(paint, tuple(points[0]), max(1, thickness // 2), 1, -1)
        cv2.circle(paint, tuple(points[-1]), max(1, thickness // 2), 1, -1)

    x0, y0, x1, y1 = box
    crop = cv2.resize(paint, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        polygons.append([[float(x + x0), float(y + y0)] for [[x, y]] in contour])
    return polygons


def buffered_line_polygon(line, width_px: float, image_shape) -> list[list[list[float]]]:
    if not line or len(line) != 2:
        return []
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    p1 = tuple(int(round(value)) for value in line[0])
    p2 = tuple(int(round(value)) for value in line[1])
    thickness = max(1, int(round(width_px)))
    cv2.line(mask, p1, p2, 1, thickness, cv2.LINE_AA)
    cv2.circle(mask, p1, max(1, thickness // 2), 1, -1)
    cv2.circle(mask, p2, max(1, thickness // 2), 1, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        [[float(x), float(y)] for [[x, y]] in contour]
        for contour in contours
        if cv2.contourArea(contour) >= 10
    ]
