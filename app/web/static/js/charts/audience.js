(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  function tok(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function fmtVal(n) {
    if (n == null || n === "") return "—";
    return String(n);
  }

  function shortDay(iso) {
    if (!iso || iso.length < 10) return iso || "";
    return iso.slice(5);
  }

  global.KizaruCharts.audience = function (el, payload, botStats) {
    if (!el) return;
    el.innerHTML = "";
    payload = payload || {};
    var strip = payload.strip || {};
    var users7 = payload.users_7d || [];
    var members7 = payload.members_7d || [];
    if (!users7.length && !strip.guilds && !strip.users_total) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }

    var wrap = document.createElement("div");
    wrap.className = "audience-wrap";

    var row = document.createElement("div");
    row.className = "audience-strip";
    var items = [
      ["Серверы", strip.guilds],
      ["Активны 15м", strip.users_active],
      ["В БД", strip.users_total],
      ["Участники", strip.members_total],
    ];
    items.forEach(function (it) {
      var cell = document.createElement("div");
      cell.className = "kpi";
      cell.innerHTML =
        '<div class="v">' +
        fmtVal(it[1]) +
        '</div><div class="l">' +
        it[0] +
        "</div>";
      row.appendChild(cell);
    });
    wrap.appendChild(row);

    if (botStats && botStats.members_online_available && strip.members_online != null) {
      var on = document.createElement("p");
      on.className = "audience-caption";
      on.textContent = "В сети Discord: " + fmtVal(strip.members_online);
      wrap.appendChild(on);
    }

    var chartHost = document.createElement("div");
    chartHost.className = "audience-chart-host";
    wrap.appendChild(chartHost);
    el.appendChild(wrap);

    if (typeof d3 === "undefined" || !users7.length) return;

    var w = chartHost.clientWidth || el.clientWidth || 480;
    var h = 132;
    var m = { t: 8, r: 12, b: 26, l: 32 };
    var fg = tok("--text");
    var series = tok("--link");
    var grid = tok("--border");
    var muted = tok("--muted");

    var data = users7.map(function (p, i) {
      return { i: i, y: Number(p.v) || 0, label: shortDay(p.t) };
    });
    var memMax = d3.max(members7, function (p) {
      return Number(p.v) || 0;
    });
    var showMembers = memMax > 0 && members7.length === users7.length;

    var svg = d3.select(chartHost).append("svg").attr("width", w).attr("height", h);
    var x = d3
      .scaleBand()
      .domain(data.map(function (d) {
        return d.i;
      }))
      .range([m.l, w - m.r])
      .padding(0.22);
    var yMax = d3.max(data, function (d) {
      return d.y;
    });
    if (showMembers) {
      yMax = Math.max(yMax, memMax);
    }
    var y = d3
      .scaleLinear()
      .domain([0, yMax || 1])
      .nice()
      .range([h - m.b, m.t]);

    svg
      .append("g")
      .attr("transform", "translate(0," + (h - m.b) + ")")
      .call(
        d3
          .axisBottom(x)
          .tickFormat(function (d) {
            return data[d] ? data[d].label : "";
          })
      )
      .attr("color", fg)
      .attr("font-size", 10);

    svg
      .append("g")
      .attr("transform", "translate(" + m.l + ",0)")
      .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format("d")))
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

    svg
      .selectAll("rect.audience-bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "audience-bar")
      .attr("x", function (d) {
        return x(d.i);
      })
      .attr("y", function (d) {
        return y(d.y);
      })
      .attr("width", x.bandwidth())
      .attr("height", function (d) {
        return h - m.b - y(d.y);
      })
      .attr("fill", series);

    if (showMembers) {
      var memData = members7.map(function (p, i) {
        return { i: i, y: Number(p.v) || 0 };
      });
      var line = d3
        .line()
        .x(function (d) {
          return x(d.i) + x.bandwidth() / 2;
        })
        .y(function (d) {
          return y(d.y);
        });
      svg
        .append("path")
        .datum(memData)
        .attr("fill", "none")
        .attr("stroke", muted)
        .attr("stroke-width", 1.5)
        .attr("d", line);
    }

    var cap = document.createElement("p");
    cap.className = "audience-caption";
    cap.textContent = showMembers
      ? "Уникальные пользователи с задачами (столбцы), участники Discord (линия), 7 дней"
      : "Уникальные пользователи с задачами, 7 дней";
    wrap.appendChild(cap);
  };
})(window);
