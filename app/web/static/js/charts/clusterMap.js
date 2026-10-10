(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  var FONT = "system-ui, \"Segoe UI\", sans-serif";
  var MONO = "ui-monospace, Consolas, monospace";
  var HEALTH_RU = { ok: "ОК", warn: "Внимание", error: "Ошибка" };

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
      canvas: tok("--map-canvas-bg") || tok("--row-alt"),
    };
  }

  function findById(list, id) {
    var hit = null;
    (list || []).forEach(function (n) {
      if (n && n.id === id) hit = n;
    });
    return hit;
  }

  function findEdge(edges, fromId, toId) {
    var e = null;
    (edges || []).forEach(function (x) {
      if (x.from === fromId && x.to === toId) e = x;
    });
    return e;
  }

  function statusOf(n) {
    if (!n) return { key: "na", label: "", stroke: null, fill: null, fg: null };
    if (n.running === false) {
      return { key: "offline", label: "Выкл", stroke: null, fill: null, fg: null };
    }
    var h = n.health || "error";
    if (h !== "ok" && h !== "warn" && h !== "error") h = "error";
    return { key: h, label: HEALTH_RU[h] || h, stroke: null, fill: null, fg: null };
  }

  function paintStatus(st, p) {
    if (st.key === "offline" || st.key === "na") {
      st.stroke = p.muted;
      st.fill = p.panel;
      st.fg = p.muted;
      return st;
    }
    if (st.key === "ok") {
      st.stroke = p.ok;
      st.fill = p.okBg;
      st.fg = p.ok;
      return st;
    }
    if (st.key === "warn") {
      st.stroke = p.warn;
      st.fill = p.warnBg;
      st.fg = p.warn;
      return st;
    }
    st.stroke = p.err;
    st.fill = p.errBg;
    st.fg = p.err;
    return st;
  }

  function tipOf(n) {
    if (!n) return "";
    var parts = [];
    if (n.label) parts.push(n.label);
    if (n.detail) parts.push(String(n.detail).replace(/^(ok|warn|error)\s*\/\s*/i, ""));
    if (n.pids) parts.push("pid " + n.pids);
    return parts.join(" / ");
  }

  function routeOf(id) {
    if (!id) return "";
    if (id === "user") return "";
    if (id === "cli") return "/jobs";
    if (id === "discord") return "/logs";
    if (id === "web") return "/";
    if (id === "orchestrator") return "/workers";
    if (id === "db" || id === "queue" || id === "sqlite") return "/system";
    if (String(id).indexOf("worker_") === 0 || id === "workers") return "/workers";
    return "";
  }

  function go(href) {
    if (href) global.location.href = href;
  }

  function trunc(s, n) {
    s = String(s || "");
    if (s.length <= n) return s;
    return s.slice(0, Math.max(0, n - 1)) + "…";
  }

  function flatWorkers(wmap) {
    var out = [];
    ["text", "mood", "pic"].forEach(function (g) {
      (wmap && wmap[g] ? wmap[g] : []).forEach(function (w) {
        out.push(w);
      });
    });
    return out;
  }

  function buildGraph(cluster) {
    var d = cluster.diagram || {};
    var edgesApi = cluster.edges || [];
    var disc = findById(d.ingress, "discord");
    var web = findById(d.ingress, "web");
    var orch = findById(d.bus, "orchestrator");
    var queue = findById(d.bus, "queue");
    var sqlite = findById(d.bus, "sqlite");
    var workers = flatWorkers(d.workers);

    var eDQ = findEdge(edgesApi, "discord", "queue");
    var eQO = findEdge(edgesApi, "queue", "orchestrator");
    var eOW = findEdge(edgesApi, "orchestrator", "workers");

    var leftX = 40;
    var boxW = 168;
    var nodes = [
      {
        id: "user",
        kind: "pill",
        label: "Пользователь Discord",
        x: 24,
        y: 28,
        w: 180,
        h: 32,
      },
      {
        id: "discord",
        kind: "box",
        label: "app.discord",
        src: disc,
        x: leftX,
        y: 88,
        w: boxW,
        h: 40,
      },
      {
        id: "cli",
        kind: "box",
        label: "app.cli",
        sub: "generate / image",
        src: null,
        x: leftX,
        y: 148,
        w: boxW,
        h: 40,
      },
      {
        id: "orchestrator",
        kind: "box",
        label: "app.orchestrator",
        src: orch,
        x: leftX,
        y: 208,
        w: boxW,
        h: 40,
      },
      {
        id: "web",
        kind: "box",
        label: "app.web",
        sub: "админка :8765",
        src: web,
        x: leftX,
        y: 268,
        w: boxW,
        h: 44,
      },
      {
        id: "db",
        kind: "hub",
        label: "DB",
        sub: (queue && queue.detail) || (sqlite && sqlite.detail) || "bot.db",
        src: queue || sqlite,
        x: 340,
        y: 168,
        w: 108,
        h: 80,
      },
    ];

    var wx = 580;
    var wy = 24;
    var ww = 280;
    var wh = 36;
    var gap = 8;
    var midIdx = workers.length ? Math.floor((workers.length - 1) / 2) : 0;

    workers.forEach(function (w, i) {
      var id = w.id || "worker_" + i;
      nodes.push({
        id: id,
        kind: "worker",
        label: w.label || w.name || id,
        src: w,
        x: wx,
        y: wy + i * (wh + gap),
        w: ww,
        h: wh,
      });
    });

    var contentH = Math.max(
      340,
      wy + workers.length * (wh + gap) + 36,
      268 + 44 + 40
    );

    var links = [
      {
        from: "user",
        to: "discord",
        label: "/ai text / mood / pic",
        count: null,
        dashed: false,
      },
      {
        from: "discord",
        to: "db",
        label: "queued",
        count: eDQ && eDQ.count != null ? eDQ.count : 0,
        dashed: false,
        yOff: -8,
      },
      {
        from: "db",
        to: "discord",
        label: "ответ",
        count: null,
        dashed: false,
        yOff: 10,
      },
      {
        from: "cli",
        to: "db",
        label: "заявки",
        count: null,
        dashed: false,
      },
      {
        from: "orchestrator",
        to: "db",
        label: "queued › assigned",
        count: eQO && eQO.count != null ? eQO.count : 0,
        dashed: false,
      },
      {
        from: "web",
        to: "db",
        label: "read",
        count: null,
        dashed: true,
      },
    ];

    workers.forEach(function (w, i) {
      var id = w.id || "worker_" + i;
      links.push({
        from: "db",
        to: id,
        label: i === midIdx ? "running › done / error" : "",
        count: i === midIdx && eOW && eOW.count != null ? eOW.count : null,
        dashed: false,
        quiet: i !== midIdx,
      });
    });

    return { nodes: nodes, links: links, contentH: contentH };
  }

  function nodeCenter(n) {
    return { x: n.x + n.w / 2, y: n.y + n.h / 2 };
  }

  function anchor(n, toward) {
    var c = nodeCenter(n);
    var dx = toward.x - c.x;
    var dy = toward.y - c.y;
    if (Math.abs(dx) >= Math.abs(dy)) {
      return { x: dx >= 0 ? n.x + n.w : n.x, y: c.y };
    }
    return { x: c.x, y: dy >= 0 ? n.y + n.h : n.y };
  }

  function drawLink(g, a, b, link, p, markerId) {
    var ta = nodeCenter(b);
    var tb = nodeCenter(a);
    var p0 = anchor(a, ta);
    var p1 = anchor(b, tb);
    var yOff = link.yOff || 0;
    p0.y += yOff;
    p1.y += yOff;
    var mx = (p0.x + p1.x) / 2;
    var my = (p0.y + p1.y) / 2;
    var path =
      "M" +
      p0.x +
      "," +
      p0.y +
      " C" +
      (p0.x + (mx - p0.x) * 0.55) +
      "," +
      p0.y +
      " " +
      (p1.x - (p1.x - mx) * 0.55) +
      "," +
      p1.y +
      " " +
      p1.x +
      "," +
      p1.y;

    var busy = link.count != null && Number(link.count) > 0;
    g.append("path")
      .attr("d", path)
      .attr("fill", "none")
      .attr("stroke", busy ? p.link : p.border)
      .attr("stroke-width", link.quiet ? 1 : busy ? 2 : 1.25)
      .attr("stroke-dasharray", link.dashed ? "5 4" : null)
      .attr("marker-end", "url(#" + markerId + ")")
      .attr("opacity", link.quiet ? 0.75 : 1);

    var label = link.label || "";
    if (link.count != null) label = (label ? label + "  " : "") + String(link.count);
    if (!label) return;

    var tw = Math.min(210, 12 + label.length * 6.2);
    g.append("rect")
      .attr("x", mx - tw / 2)
      .attr("y", my - 10)
      .attr("width", tw)
      .attr("height", 18)
      .attr("rx", 3)
      .attr("fill", p.panel)
      .attr("stroke", p.border);
    g.append("text")
      .attr("x", mx)
      .attr("y", my + 3)
      .attr("text-anchor", "middle")
      .attr("fill", p.muted)
      .attr("font-size", 10)
      .attr("font-family", FONT)
      .text(trunc(label, 30));
  }

  function drawBadge(el, n, st, p) {
    if (!st.label) return;
    var bw = Math.max(28, st.label.length * 6.6 + 10);
    el.append("rect")
      .attr("x", n.w - bw - 8)
      .attr("y", (n.h - 18) / 2)
      .attr("width", bw)
      .attr("height", 18)
      .attr("rx", 3)
      .attr("fill", p.panel)
      .attr("stroke", st.stroke);
    el.append("text")
      .attr("x", n.w - bw / 2 - 8)
      .attr("y", n.h / 2 + 4)
      .attr("text-anchor", "middle")
      .attr("fill", st.fg)
      .attr("font-size", 10)
      .attr("font-weight", 700)
      .attr("font-family", FONT)
      .text(st.label);
  }

  function drawNode(g, n, p) {
    var src = n.src;
    var st = paintStatus(statusOf(src), p);
    var href = routeOf(n.id);
    var el = g
      .append("g")
      .attr("class", "arch-node")
      .attr("transform", "translate(" + n.x + "," + n.y + ")")
      .attr("tabindex", href ? 0 : null)
      .style("cursor", href ? "pointer" : "default");

    var tip = tipOf(src) || n.sub || n.label;
    if (st.label) tip = (tip ? tip + " / " : "") + st.label;
    el.append("title").text(tip);

    if (n.kind === "pill") {
      el.append("rect")
        .attr("width", n.w)
        .attr("height", n.h)
        .attr("rx", 3)
        .attr("fill", p.panel)
        .attr("stroke", p.link)
        .attr("stroke-width", 1.5);
      el.append("text")
        .attr("x", n.w / 2)
        .attr("y", n.h / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("fill", p.text)
        .attr("font-size", 12)
        .attr("font-weight", 600)
        .attr("font-family", FONT)
        .text(n.label);
      return;
    }

    var fill = n.kind === "hub" ? p.rowAlt : st.fill || p.panel;
    var stroke = n.kind === "hub" ? healthStrokeHub(src, p) : st.stroke || p.border;

    el.append("rect")
      .attr("width", n.w)
      .attr("height", n.h)
      .attr("rx", 3)
      .attr("fill", fill)
      .attr("stroke", stroke)
      .attr("stroke-width", n.kind === "hub" ? 2 : 1.5);

    if (n.kind === "hub") {
      [
        [10, 10],
        [n.w - 10, 10],
        [10, n.h - 10],
        [n.w - 10, n.h - 10],
      ].forEach(function (pt) {
        el.append("circle")
          .attr("cx", pt[0])
          .attr("cy", pt[1])
          .attr("r", 3)
          .attr("fill", p.link);
      });
      el.append("text")
        .attr("x", n.w / 2)
        .attr("y", n.h / 2 - 2)
        .attr("text-anchor", "middle")
        .attr("fill", p.text)
        .attr("font-size", 18)
        .attr("font-weight", 700)
        .attr("font-family", MONO)
        .text(n.label);
      if (n.sub) {
        el.append("text")
          .attr("x", n.w / 2)
          .attr("y", n.h / 2 + 16)
          .attr("text-anchor", "middle")
          .attr("fill", p.muted)
          .attr("font-size", 10)
          .attr("font-family", FONT)
          .text(trunc(n.sub, 14));
      }
    } else if (n.kind === "worker") {
      el.append("circle")
        .attr("cx", 12)
        .attr("cy", n.h / 2)
        .attr("r", 4)
        .attr("fill", st.stroke || p.muted);
      el.append("text")
        .attr("x", 22)
        .attr("y", n.h / 2 + 4)
        .attr("fill", p.text)
        .attr("font-size", 12)
        .attr("font-weight", 700)
        .attr("font-family", FONT)
        .text(trunc(n.label, 18));
      drawBadge(el, n, st, p);
    } else {
      el.append("text")
        .attr("x", 10)
        .attr("y", n.sub ? 16 : n.h / 2 + 4)
        .attr("fill", p.text)
        .attr("font-size", 12)
        .attr("font-weight", 700)
        .attr("font-family", FONT)
        .text(trunc(n.label, 16));
      if (n.sub) {
        el.append("text")
          .attr("x", 10)
          .attr("y", 32)
          .attr("fill", p.muted)
          .attr("font-size", 10)
          .attr("font-family", FONT)
          .text(trunc(n.sub, 22));
      }
      if (src) drawBadge(el, n, st, p);
    }

    if (href) {
      el.on("click", function () {
        go(href);
      });
      el.on("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          go(href);
        }
      });
    }
  }

  function healthStrokeHub(src, p) {
    var st = paintStatus(statusOf(src), p);
    return st.stroke || p.border;
  }

  global.KizaruCharts.clusterMap = function (el, cluster) {
    if (!el) return;
    el.innerHTML = "";
    cluster = cluster || {};
    if (!cluster.diagram) {
      el.innerHTML = '<p class="muted">нет схемы</p>';
      return;
    }
    if (typeof d3 === "undefined") {
      el.innerHTML = '<p class="muted">нет d3</p>';
      return;
    }
    var p = palette();
    if (!p.panel || !p.border) return;

    var graph = buildGraph(cluster);
    var byId = {};
    graph.nodes.forEach(function (n) {
      byId[n.id] = n;
    });

    var wrap = d3
      .select(el)
      .append("div")
      .attr("class", "cluster-map")
      .attr("role", "img")
      .attr(
        "aria-label",
        "Схема сервисов и всех воркеров: связи через DB, статусы на блоках"
      );

    var pad = 12;
    var baseW = 900;
    var hostW = el.clientWidth || baseW;
    var svgW = Math.max(baseW, Math.min(1200, hostW - 8));
    var scaleX = svgW / baseW;
    if (scaleX > 1.02) {
      graph.nodes.forEach(function (n) {
        if (n.x >= 300) n.x = Math.round(n.x * scaleX);
        if (n.kind === "worker") n.w = Math.round(n.w * Math.min(1.2, scaleX));
      });
    }

    var svgH = Math.max(400, graph.contentH + pad * 2);

    var svg = wrap
      .append("svg")
      .attr("width", svgW)
      .attr("height", svgH)
      .attr("viewBox", "0 0 " + svgW + " " + svgH);

    var defs = svg.append("defs");
    var mid = "archArrow";
    defs
      .append("marker")
      .attr("id", mid)
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,0 L10,5 L0,10 Z")
      .attr("fill", p.border);

    svg
      .append("rect")
      .attr("width", svgW)
      .attr("height", svgH)
      .attr("fill", p.canvas || p.rowAlt);

    var root = svg.append("g").attr("transform", "translate(" + pad + "," + pad + ")");

    root
      .append("text")
      .attr("x", 580 * Math.min(scaleX, 1.2))
      .attr("y", 12)
      .attr("fill", p.muted)
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("font-family", FONT)
      .text("воркеры");

    var linkG = root.append("g").attr("class", "arch-links");
    graph.links.forEach(function (link) {
      var a = byId[link.from];
      var b = byId[link.to];
      if (!a || !b) return;
      drawLink(linkG, a, b, link, p, mid);
    });

    var nodeG = root.append("g").attr("class", "arch-nodes");
    graph.nodes.forEach(function (n) {
      drawNode(nodeG, n, p);
    });

    root
      .append("text")
      .attr("x", 0)
      .attr("y", svgH - pad - 6)
      .attr("fill", p.muted)
      .attr("font-size", 11)
      .attr("font-family", FONT)
      .text("Шина = SQLite bot.db  /  каждый воркер = блок со статусом");
  };
})(window);
