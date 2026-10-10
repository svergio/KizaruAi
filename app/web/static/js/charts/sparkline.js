(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  global.KizaruCharts.sparkline = function (el, points) {
    if (!el) return;
    el.innerHTML = "";
    if (!points || !points.length) {
      el.innerHTML = '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;
    var w = el.clientWidth || 240;
    var h = 48;
    var data = points.map(function (p, i) { return { i: i, v: Number(p.v) || 0 }; });
    var x = d3.scaleLinear().domain([0, data.length - 1]).range([2, w - 2]);
    var y = d3.scaleLinear().domain([0, d3.max(data, function (d) { return d.v; }) || 1]).range([h - 2, 2]);
    var line = d3.line().x(function (d) { return x(d.i); }).y(function (d) { return y(d.v); });
    d3.select(el).append("svg").attr("width", w).attr("height", h)
      .append("path").datum(data).attr("fill", "none")
      .attr("stroke", color("--link"))
      .attr("stroke-width", 1.5).attr("d", line);
  };
})(window);
