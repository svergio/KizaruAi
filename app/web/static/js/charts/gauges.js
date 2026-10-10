(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function palette() {
    return {
      panel: tok("--panel"),
      border: tok("--border"),
      text: tok("--text"),
      muted: tok("--muted"),
      link: tok("--link"),
      rowAlt: tok("--row-alt"),
      ok: tok("--status-done-fg"),
      warn: tok("--warn-fg"),
      err: tok("--status-error-fg"),
      okBg: tok("--status-done-bg"),
      warnBg: tok("--warn-bg"),
      errBg: tok("--status-error-bg"),
    };
  }

  function levelColor(p, level) {
    if (level === "ok") return p.ok;
    if (level === "warn") return p.warn;
    if (level === "error") return p.err;
    return p.muted;
  }

  function drawOne(svg, cx, cy, r, item, p) {
    var pct = item.pct;
    var level = item.level || "na";
    var na = level === "na" || pct == null || isNaN(Number(pct));
    var v = na ? 0 : Math.max(0, Math.min(100, Number(pct)));
    var col = levelColor(p, level);

    var track = d3
      .arc()
      .innerRadius(r - 14)
      .outerRadius(r)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2);

    var g = svg.append("g").attr("transform", "translate(" + cx + "," + cy + ")");

    g.append("path").attr("d", track()).attr("fill", p.rowAlt).attr("stroke", p.border);

    var warnStart = -Math.PI / 2 + Math.PI * 0.7;
    var errStart = -Math.PI / 2 + Math.PI * 0.9;
    var band = d3.arc().innerRadius(r - 14).outerRadius(r);
    g.append("path")
      .attr(
        "d",
        band({ startAngle: warnStart, endAngle: errStart })
      )
      .attr("fill", p.warn)
      .attr("opacity", 0.35);
    g.append("path")
      .attr(
        "d",
        band({ startAngle: errStart, endAngle: Math.PI / 2 })
      )
      .attr("fill", p.err)
      .attr("opacity", 0.4);

    if (!na && v > 0) {
      var end = -Math.PI / 2 + Math.PI * (v / 100);
      var valueArc = d3
        .arc()
        .innerRadius(r - 14)
        .outerRadius(r)
        .startAngle(-Math.PI / 2)
        .endAngle(end);
      g.append("path").attr("d", valueArc()).attr("fill", col);
    }

    var theta = -Math.PI / 2 + Math.PI * (v / 100);
    var needleR = r - 18;
    var tipX = Math.sin(theta) * needleR;
    var tipY = -Math.cos(theta) * needleR;
    g.append("line")
      .attr("x1", 0)
      .attr("y1", 0)
      .attr("x2", tipX)
      .attr("y2", tipY)
      .attr("stroke", na ? p.muted : col)
      .attr("stroke-width", 2.5)
      .attr("stroke-linecap", "round");
    g.append("circle").attr("r", 5).attr("fill", p.panel).attr("stroke", na ? p.muted : col).attr("stroke-width", 2);

    var label = item.label || "";
    var valueText = na ? "н/д" : Math.round(v) + "%";
    var valueY = -Math.round(r * 0.4);
    var labelY = 14;
    g.append("text")
      .attr("class", "gauge-value")
      .attr("y", valueY)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("fill", p.text)
      .text(valueText);
    g.append("text")
      .attr("class", "gauge-label")
      .attr("y", labelY)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "hanging")
      .attr("fill", p.muted)
      .text(label);
  }

  global.KizaruCharts.gauges = function (el, bars) {
    if (!el) return;
    el.innerHTML = "";
    if (typeof d3 === "undefined") {
      el.innerHTML = '<div class="empty">нет d3</div>';
      return;
    }
    var p = palette();
    if (!p.panel || !p.border) return;
    bars = bars || {};
    var items = [
      { key: "queue", label: "Очередь", bar: bars.queue },
      { key: "cpu", label: "CPU", bar: bars.cpu },
      { key: "mem", label: "RAM", bar: bars.mem },
      { key: "gpu", label: "GPU", bar: bars.gpu },
    ].map(function (it) {
      var b = it.bar || {};
      return {
        label: it.label,
        pct: b.pct,
        level: b.level || "na",
      };
    });

    var hostW = el.clientWidth || 640;
    var cellW = Math.min(200, Math.max(140, Math.floor(hostW / items.length)));
    var r = Math.min(64, cellW * 0.38);
    var padTop = 6;
    var labelBand = 14 + 14;
    var padBottom = 10;
    var cy = r + padTop;
    var h = cy + labelBand + padBottom;
    var svgW = cellW * items.length;
    var svg = d3
      .select(el)
      .append("svg")
      .attr("class", "gauges-svg")
      .attr("width", svgW)
      .attr("height", h)
      .attr("viewBox", "0 0 " + svgW + " " + h)
      .attr("role", "img")
      .attr("aria-label", "Нагрузка: очередь, CPU, RAM, GPU");

    items.forEach(function (item, i) {
      var cx = cellW * i + cellW / 2;
      drawOne(svg, cx, cy, r, item, p);
    });

    var gpuDetail = (bars.gpu && bars.gpu.detail) || [];
    if (gpuDetail.length) {
      var vramH = 14 + gpuDetail.length * 18;
      var vramSvg = d3
        .select(el)
        .append("svg")
        .attr("class", "gpu-vram-svg")
        .attr("width", svgW)
        .attr("height", vramH)
        .attr("viewBox", "0 0 " + svgW + " " + vramH)
        .attr("role", "img")
        .attr("aria-label", "VRAM по GPU, MiB");
      var trackW = Math.min(svgW - 24, 520);
      var tx = (svgW - trackW) / 2;
      gpuDetail.forEach(function (g, idx) {
        var used = Number(g.mem_used_mib) || 0;
        var total = Number(g.mem_total_mib) || 0;
        var pct = total > 0 ? used / total : 0;
        var y0 = 8 + idx * 18;
        var label = g.name || "GPU " + idx;
        vramSvg
          .append("text")
          .attr("x", tx)
          .attr("y", y0 + 10)
          .attr("fill", p.muted)
          .attr("font-size", 10)
          .text(label + " " + used + "/" + total + " MiB");
        vramSvg
          .append("rect")
          .attr("x", tx)
          .attr("y", y0 + 12)
          .attr("width", trackW)
          .attr("height", 6)
          .attr("fill", p.rowAlt)
          .attr("stroke", p.border);
        vramSvg
          .append("rect")
          .attr("x", tx)
          .attr("y", y0 + 12)
          .attr("width", Math.max(0, trackW * pct))
          .attr("height", 6)
          .attr("fill", p.link);
      });
    }
  };
})(window);
