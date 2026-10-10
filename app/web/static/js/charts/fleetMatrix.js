(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function workerState(w) {
    if (!w || !w.running) return "offline";
    if (w.health === "error") return "failing";
    if (w.health === "warn") return "idle";
    return "online";
  }

  function cellColor(state, p) {
    if (state === "online") return p.ok;
    if (state === "idle") return p.warn;
    if (state === "failing") return p.err;
    return p.muted;
  }

  global.KizaruCharts.fleetMatrix = function (el, workers, models, ckptByWorker) {
    if (!el) return;
    el.innerHTML = "";
    workers = workers || [];
    models = models || [];
    ckptByWorker = ckptByWorker || {};
    var prod = models.filter(function (m) {
      return m.present !== false;
    });
    if (!workers.length || !prod.length) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    if (typeof d3 === "undefined") return;

    var p = {
      text: tok("--text"),
      muted: tok("--muted"),
      border: tok("--border"),
      panel: tok("--panel"),
      link: tok("--link"),
      ok: tok("--status-done-fg"),
      warn: tok("--warn"),
      err: tok("--status-error-fg"),
      mutedCell: tok("--row-alt"),
    };

    var labelW = 120;
    var colW = 52;
    var rowH = 22;
    var headH = 48;
    var w = labelW + prod.length * colW + 12;
    var h = headH + workers.length * rowH + 16;
    var svg = d3
      .select(el)
      .append("svg")
      .attr("width", w)
      .attr("height", h)
      .attr("role", "img")
      .attr("aria-label", "Матрица воркеров и prod-моделей");

    prod.forEach(function (m, i) {
      var name = (m.label || m.name || "").replace(".pt", "");
      svg
        .append("text")
        .attr("x", labelW + i * colW + colW / 2)
        .attr("y", 12)
        .attr("text-anchor", "end")
        .attr("transform", "rotate(-42 " + (labelW + i * colW + colW / 2) + ",12)")
        .attr("fill", p.muted)
        .attr("font-size", 9)
        .text(name.length > 10 ? name.slice(0, 9) + "…" : name);
    });

    workers.forEach(function (wkr, ri) {
      var y = headH + ri * rowH;
      var wk = wkr.worker_kind || wkr.key || "";
      var state = workerState(wkr);
      svg
        .append("text")
        .attr("x", 4)
        .attr("y", y + 15)
        .attr("fill", state === "offline" ? p.muted : p.text)
        .attr("font-size", 11)
        .text(wkr.key || wk);
      var ckpt = ckptByWorker[wk] || "";
      prod.forEach(function (m, ci) {
        var x = labelW + ci * colW;
        var match = ckpt && m.name === ckpt;
        var fill = match ? cellColor(state, p) : p.mutedCell;
        var op = match ? (state === "offline" ? 0.35 : 0.9) : 0.08;
        svg
          .append("rect")
          .attr("x", x + 2)
          .attr("y", y + 2)
          .attr("width", colW - 4)
          .attr("height", rowH - 4)
          .attr("fill", fill)
          .attr("opacity", op)
          .attr("stroke", match ? fill : p.border)
          .attr("stroke-width", match ? 1 : 0.5)
          .append("title")
          .text(
            (wkr.key || wk) +
              " / " +
              (m.name || "") +
              (match ? " (" + state + ")" : "")
          );
      });
    });
  };
})(window);
