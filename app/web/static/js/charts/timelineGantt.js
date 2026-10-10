(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  global.KizaruCharts.timelineGantt = function (el, items) {
    if (!el) return;
    el.innerHTML = "";
    var rows = (items || []).filter(function (d) { return d.start; }).slice(0, 40);
    if (!rows.length) {
      el.innerHTML = '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;
    var parsed = rows.map(function (d) {
      var s = new Date(d.start);
      var e = new Date(d.end || d.start);
      if (isNaN(e.getTime()) || e < s) e = new Date(s.getTime() + 1000);
      return { label: d.label || String(d.id), start: s, end: e, status: d.status };
    });
    var w = el.clientWidth || 480;
    var rowH = 18;
    var m = { t: 8, r: 8, b: 24, l: 56 };
    var h = m.t + m.b + parsed.length * rowH;
    var x = d3.scaleTime().domain([
      d3.min(parsed, function (d) { return d.start; }),
      d3.max(parsed, function (d) { return d.end; }),
    ]).range([m.l, w - m.r]);
    var svg = d3.select(el).append("svg").attr("width", w).attr("height", h);
    var fg = color("--text");
    var statusFill = {
      queued: color("--status-queued-bg"),
      assigned: color("--status-assigned-bg"),
      running: color("--status-running-bg"),
      done: color("--status-done-bg"),
      error: color("--status-error-bg"),
      cancelled: color("--status-cancelled-bg"),
    };
    svg.append("g").attr("transform", "translate(0," + (h - m.b) + ")")
      .call(d3.axisBottom(x).ticks(5)).attr("color", fg).attr("font-size", 10);
    parsed.forEach(function (d, i) {
      var y = m.t + i * rowH;
      svg.append("text").attr("x", 4).attr("y", y + 12).attr("fill", fg).attr("font-size", 10).text(d.label);
      svg.append("rect")
        .attr("x", x(d.start))
        .attr("y", y + 3)
        .attr("width", Math.max(2, x(d.end) - x(d.start)))
        .attr("height", 12)
        .attr("fill", statusFill[d.status] || color("--muted"));
    });
  };
})(window);
