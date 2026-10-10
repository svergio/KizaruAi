(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function statusColor(key, which) {
    var v = tok("--status-" + key + "-" + which);
    return v || tok(which === "fg" ? "--text" : "--row-alt");
  }

  global.KizaruCharts.statusDonut = function (el, byStatus) {
    if (!el) return;
    el.innerHTML = "";
    byStatus = byStatus || {};
    var statusMap = global.KIZARU_STATUS || {};
    var entries = Object.keys(statusMap)
      .map(function (k) {
        return { key: k, v: Number(byStatus[k]) || 0, label: statusMap[k].label || k };
      })
      .filter(function (d) {
        return d.v > 0;
      });
    if (!entries.length) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;

    var w = el.clientWidth || 220;
    var h = 180;
    var r = Math.min(72, w / 2 - 16);
    var svg = d3
      .select(el)
      .append("svg")
      .attr("width", w)
      .attr("height", h)
      .attr("role", "img")
      .attr("aria-label", "Распределение задач по статусам");

    var g = svg.append("g").attr("transform", "translate(" + w / 2 + "," + (r + 18) + ")");
    var pie = d3
      .pie()
      .sort(null)
      .value(function (d) {
        return d.v;
      });
    var arc = d3
      .arc()
      .innerRadius(r * 0.52)
      .outerRadius(r);
    var data = pie(entries);
    var total = d3.sum(entries, function (d) {
      return d.v;
    });

    g.selectAll("path")
      .data(data)
      .enter()
      .append("path")
      .attr("d", arc)
      .attr("fill", function (d) {
        return statusColor(d.data.key, "bg");
      })
      .attr("stroke", tok("--panel"))
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .append("title")
      .text(function (d) {
        return d.data.label + ": " + d.data.v;
      });

    g.selectAll("path")
      .on("click", function (_ev, d) {
        global.location.href = "/jobs?status=" + encodeURIComponent(d.data.key) + "&page=1";
      });

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("y", 6)
      .attr("fill", tok("--text"))
      .attr("font-size", 16)
      .attr("font-weight", 700)
      .text(String(total));

    var legend = svg.append("g").attr("transform", "translate(8," + (r * 2 + 28) + ")");
    var lx = 0;
    var ly = 0;
    entries.forEach(function (d) {
      var label = d.label + " " + d.v;
      var lw = label.length * 6.5 + 16;
      if (lx + lw > w - 8) {
        lx = 0;
        ly += 16;
      }
      legend
        .append("rect")
        .attr("x", lx)
        .attr("y", ly)
        .attr("width", 8)
        .attr("height", 8)
        .attr("fill", statusColor(d.key, "fg"));
      legend
        .append("text")
        .attr("x", lx + 12)
        .attr("y", ly + 8)
        .attr("fill", tok("--muted"))
        .attr("font-size", 10)
        .text(label);
      lx += lw + 8;
    });
  };
})(window);
