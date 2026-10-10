(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  var DAY_LABELS = ["6д назад", "5д", "4д", "3д", "2д", "1д", "сегодня"];

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  global.KizaruCharts.heatmap = function (el, payload) {
    if (!el) return;
    el.innerHTML = "";
    var cells = (payload && payload.cells) || [];
    if (!cells.length) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;
    var rows = (payload && payload.rows) || 7;
    var cols = (payload && payload.cols) || 24;
    var w = el.clientWidth || 480;
    var m = { t: 8, r: 8, b: 22, l: 36 };
    var innerW = w - m.l - m.r;
    var innerH = Math.max(100, rows * 14);
    var h = innerH + m.t + m.b;
    var max = d3.max(cells, function (c) {
      return Number(c.v) || 0;
    });
    if (!max) max = 1;
    var cellW = innerW / cols;
    var cellH = innerH / rows;
    var link = tok("--link");
    var muted = tok("--muted");
    var border = tok("--border");
    if (!link) return;

    var svg = d3
      .select(el)
      .append("svg")
      .attr("width", w)
      .attr("height", h + 20)
      .attr("role", "img")
      .attr("aria-label", "Тепловая карта создания задач: 7 дней по 24 часа UTC");

    var g = svg.append("g").attr("transform", "translate(" + m.l + "," + m.t + ")");

    g.selectAll("rect")
      .data(cells)
      .enter()
      .append("rect")
      .attr("x", function (d) {
        return d.h * cellW;
      })
      .attr("y", function (d) {
        return d.d * cellH;
      })
      .attr("width", Math.max(1, cellW - 1))
      .attr("height", Math.max(1, cellH - 1))
      .attr("fill", link)
      .attr("stroke", border)
      .attr("stroke-width", 0.5)
      .attr("opacity", function (d) {
        var v = Number(d.v) || 0;
        return v <= 0 ? 0.06 : 0.15 + 0.85 * (v / max);
      })
      .append("title")
      .text(function (d) {
        var day = DAY_LABELS[d.d] || String(d.d);
        return day + ", " + d.h + ":00 — " + (Number(d.v) || 0) + " задач";
      });

    for (var hi = 0; hi < cols; hi += 6) {
      g.append("text")
        .attr("x", hi * cellW + 2)
        .attr("y", innerH + 14)
        .attr("fill", muted)
        .attr("font-size", 9)
        .text(String(hi));
    }

    DAY_LABELS.forEach(function (lab, di) {
      if (di % 2 !== 0 && di !== rows - 1) return;
      g.append("text")
        .attr("x", -4)
        .attr("y", di * cellH + cellH * 0.72)
        .attr("text-anchor", "end")
        .attr("fill", muted)
        .attr("font-size", 9)
        .text(lab);
    });

    var legend = svg.append("g").attr("transform", "translate(" + m.l + "," + (h + 4) + ")");
    legend
      .append("text")
      .attr("x", 0)
      .attr("y", 10)
      .attr("fill", muted)
      .attr("font-size", 10)
      .text("меньше");
    var legW = 80;
    for (var i = 0; i <= 4; i++) {
      legend
        .append("rect")
        .attr("x", 42 + i * (legW / 4))
        .attr("y", 2)
        .attr("width", legW / 4 - 1)
        .attr("height", 8)
        .attr("fill", link)
        .attr("opacity", 0.1 + (0.85 * i) / 4);
    }
    legend
      .append("text")
      .attr("x", 42 + legW + 6)
      .attr("y", 10)
      .attr("fill", muted)
      .attr("font-size", 10)
      .text("больше (макс " + max + ")");
  };
})(window);
