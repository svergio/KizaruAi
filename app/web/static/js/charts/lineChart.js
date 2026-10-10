(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function color(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  global.KizaruCharts.lineChart = function (el, points, opts) {
    if (!el) return;
    opts = opts || {};
    el.innerHTML = "";
    if (!points || !points.length) {
      el.innerHTML = '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;
    var w = el.clientWidth || 480;
    var h = 160;
    var m = { t: 10, r: 12, b: 28, l: 36 };
    var svg = d3.select(el).append("svg").attr("width", w).attr("height", h);
    var data = points.map(function (p, i) {
      return {
        x: opts.categorical ? i : new Date(p.t),
        y: Number(p.v) || 0,
        label: p.t,
      };
    });
    var x = opts.categorical
      ? d3.scaleBand().domain(data.map(function (d) { return d.x; })).range([m.l, w - m.r]).padding(0.2)
      : d3.scaleTime().domain(d3.extent(data, function (d) { return d.x; })).range([m.l, w - m.r]);
    var y = d3.scaleLinear().domain([0, d3.max(data, function (d) { return d.y; }) || 1]).nice().range([h - m.b, m.t]);
    var fg = color("--text");
    var series = color("--link");
    var grid = color("--border");
    svg.append("g").attr("transform", "translate(0," + (h - m.b) + ")").call(d3.axisBottom(x).ticks(5)).attr("color", fg).attr("font-size", 10);
    svg.append("g").attr("transform", "translate(" + m.l + ",0)").call(d3.axisLeft(y).ticks(4)).attr("color", fg).attr("font-size", 10);
    svg.append("g").attr("stroke", grid).attr("stroke-opacity", 0.4)
      .call(d3.axisLeft(y).ticks(4).tickSize(-(w - m.l - m.r)).tickFormat(""));
    if (opts.categorical) {
      svg.selectAll("rect").data(data).enter().append("rect")
        .attr("x", function (d) { return x(d.x); })
        .attr("y", function (d) { return y(d.y); })
        .attr("width", x.bandwidth())
        .attr("height", function (d) { return h - m.b - y(d.y); })
        .attr("fill", series);
    } else {
      var line = d3.line().x(function (d) { return x(d.x); }).y(function (d) { return y(d.y); });
      svg.append("path").datum(data).attr("fill", "none").attr("stroke", series).attr("stroke-width", 1.5).attr("d", line);
    }
  };

  global.KizaruCharts.timeseriesHome = function (el, seriesMap) {
    if (!el) return;
    el.innerHTML = "";
    seriesMap = seriesMap || {};
    var keys = ["jobs_created", "jobs_done", "jobs_error"];
    var labels = {
      jobs_created: "создано",
      jobs_done: "готово",
      jobs_error: "ошибки",
    };
    var colors = {
      jobs_created: color("--link"),
      jobs_done: color("--status-done-fg"),
      jobs_error: color("--status-error-fg"),
    };
    var merged = [];
    keys.forEach(function (k) {
      (seriesMap[k] || []).forEach(function (p) {
        merged.push({ t: p.t, y: Number(p.v) || 0, key: k });
      });
    });
    if (!merged.length) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;
    var w = el.clientWidth || 480;
    var h = 180;
    var m = { t: 12, r: 12, b: 28, l: 36 };
    var svg = d3
      .select(el)
      .append("svg")
      .attr("width", w)
      .attr("height", h)
      .attr("role", "img")
      .attr("aria-label", "Динамика задач: создано, готово, ошибки");
    var xs = [];
    keys.forEach(function (k) {
      (seriesMap[k] || []).forEach(function (p) {
        xs.push(new Date(p.t));
      });
    });
    var x = d3
      .scaleTime()
      .domain(d3.extent(xs))
      .range([m.l, w - m.r]);
    var yMax =
      d3.max(merged, function (d) {
        return d.y;
      }) || 1;
    var y = d3.scaleLinear().domain([0, yMax]).nice().range([h - m.b, m.t]);
    var fg = color("--text");
    var grid = color("--border");
    svg
      .append("g")
      .attr("transform", "translate(0," + (h - m.b) + ")")
      .call(d3.axisBottom(x).ticks(5))
      .attr("color", fg)
      .attr("font-size", 10);
    svg
      .append("g")
      .attr("transform", "translate(" + m.l + ",0)")
      .call(d3.axisLeft(y).ticks(4))
      .attr("color", fg)
      .attr("font-size", 10);
    svg
      .append("g")
      .attr("stroke", grid)
      .attr("stroke-opacity", 0.4)
      .call(
        d3
          .axisLeft(y)
          .ticks(4)
          .tickSize(-(w - m.l - m.r))
          .tickFormat("")
      );
    keys.forEach(function (k) {
      var pts = (seriesMap[k] || []).map(function (p) {
        return { x: new Date(p.t), y: Number(p.v) || 0 };
      });
      if (!pts.length) return;
      var line = d3
        .line()
        .x(function (d) {
          return x(d.x);
        })
        .y(function (d) {
          return y(d.y);
        });
      svg
        .append("path")
        .datum(pts)
        .attr("fill", "none")
        .attr("stroke", colors[k] || color("--link"))
        .attr("stroke-width", 1.5)
        .attr("d", line);
    });
    var lx = m.l;
    keys.forEach(function (k) {
      if (!(seriesMap[k] || []).length) return;
      svg
        .append("line")
        .attr("x1", lx)
        .attr("y1", 8)
        .attr("x2", lx + 14)
        .attr("y2", 8)
        .attr("stroke", colors[k])
        .attr("stroke-width", 2);
      svg
        .append("text")
        .attr("x", lx + 18)
        .attr("y", 11)
        .attr("fill", fg)
        .attr("font-size", 10)
        .text(labels[k] || k);
      lx += 72;
    });
  };
})(window);
