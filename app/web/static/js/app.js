(function () {
  "use strict";

  var PAUSE_KEY = "kizaru_admin_pause";
  var MSK = "Europe/Moscow";
  var main = document.getElementById("main");
  if (!main) return;
  var page = main.getAttribute("data-page") || "home";
  var refreshSec = parseInt(main.getAttribute("data-refresh") || "3", 10) || 3;
  var paused = localStorage.getItem(PAUSE_KEY) === "1";
  var timer = null;
  var csrfToken = "";
  var homeTsRange = "24h";

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  var MONTHS_RU = ["", "янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];

  function fmtMsk(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    var parts = new Intl.DateTimeFormat("ru-RU", {
      timeZone: MSK,
      day: "numeric",
      month: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).formatToParts(d);
    var map = {};
    parts.forEach(function (p) {
      map[p.type] = p.value;
    });
    var hh = map.hour + ":" + map.minute + ":" + map.second;
    var todayParts = new Intl.DateTimeFormat("en-CA", {
      timeZone: MSK,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(new Date());
    var tmap = {};
    todayParts.forEach(function (p) {
      tmap[p.type] = p.value;
    });
    var sameDay =
      String(parseInt(map.day, 10)) === String(parseInt(tmap.day, 10)) &&
      String(parseInt(map.month, 10)) === String(parseInt(tmap.month, 10)) &&
      map.year === tmap.year;
    if (sameDay) return hh;
    var mon = MONTHS_RU[parseInt(map.month, 10)] || map.month;
    return parseInt(map.day, 10) + " " + mon + ", " + hh;
  }

  function maskDiscordId(value) {
    if (value == null) return "—";
    var s = String(value);
    if (s === "cli" || s.indexOf("cli:") === 0) return s;
    if (s.length <= 8) return s;
    return s.slice(0, 4) + "…" + s.slice(-2);
  }

  function fmtDur(sec) {
    if (sec == null || isNaN(sec)) return "—";
    var s = Number(sec);
    if (s < 0) s = 0;
    if (s < 1) return Math.round(s * 1000) + " мс";
    if (s < 60) {
      var t = Math.round(s * 10) / 10;
      return String(t).replace(".", ",") + " с";
    }
    var total = Math.round(s);
    var h = Math.floor(total / 3600);
    var m = Math.floor((total % 3600) / 60);
    var secPart = total % 60;
    if (h > 0) {
      return h + " ч " + String(m).padStart(2, "0") + " мин";
    }
    return m + " мин " + secPart + " с";
  }

  function truncEllipsis(text, maxLen) {
    if (text == null) return "";
    var s = String(text);
    if (s.length <= maxLen) return s;
    if (maxLen <= 1) return "…";
    return s.slice(0, maxLen - 1) + "…";
  }

  function userLabel(id, name) {
    if (name) return name + " (" + id + ")";
    return id || "—";
  }

  function avatarHtml(url, cls) {
    if (!url) return "";
    return (
      '<img class="' +
      (cls || "avatar") +
      '" src="' +
      escapeHtml(url) +
      '" alt="" width="' +
      (cls === "avatar-lg" ? "48" : "16") +
      '" height="' +
      (cls === "avatar-lg" ? "48" : "16") +
      '">'
    );
  }

  // Expose for tests / console.
  window.KizaruFmt = {
    fmtMsk: fmtMsk,
    fmtDur: fmtDur,
    truncEllipsis: truncEllipsis,
    maskDiscordId: maskDiscordId,
  };

  function kindLabel(k) {
    return (window.KIZARU_KIND && window.KIZARU_KIND[k]) || k || "—";
  }

  function workerLabel(k) {
    return (window.KIZARU_WORKER_KIND && window.KIZARU_WORKER_KIND[k]) || k || "—";
  }

  function statusChip(status) {
    var meta = (window.KIZARU_STATUS && window.KIZARU_STATUS[status]) || {};
    var cls = meta.css || "status-queued";
    var label = meta.label || status;
    return '<span class="badge ' + cls + '">' + escapeHtml(label) + "</span>";
  }

  var STAGE_LABELS = {
    created: "Создана",
    started: "Старт",
    finished: "Финиш",
  };

  function kindBlob(kind) {
    return (
      '<span class="job-blob kind-blob">' + escapeHtml(kindLabel(kind)) + "</span>"
    );
  }

  function stageTrail(stages) {
    var list = stages || [];
    if (!list.length) return '<span class="muted">—</span>';
    var parts = [];
    list.forEach(function (s, i) {
      var done = !!s.at;
      var label = STAGE_LABELS[s.key] || s.key || "—";
      var cls = "job-blob stage-blob " + (done ? "stage-done" : "stage-pending");
      var title = done ? ' title="' + escapeHtml(fmtMsk(s.at)) + '"' : "";
      parts.push("<span class=\"" + cls + "\"" + title + ">" + escapeHtml(label) + "</span>");
      if (i < list.length - 1) {
        parts.push('<span class="job-trail-sep" aria-hidden="true">›</span>');
      }
    });
    return '<span class="job-trail">' + parts.join("") + "</span>";
  }

  function homeJobLogRow(j) {
    var href = "/jobs/" + j.id;
    return (
      '<a class="job-log-row" href="' +
      href +
      '">' +
      '<span class="job-log-id num">#' +
      j.id +
      "</span>" +
      '<span class="job-log-who">' +
      escapeHtml(j.user_mask || "—") +
      "</span>" +
      '<span class="job-log-kind">' +
      kindBlob(j.kind) +
      "</span>" +
      '<span class="job-log-status">' +
      statusChip(j.status) +
      "</span>" +
      '<span class="job-log-stages">' +
      stageTrail(j.stages) +
      "</span>" +
      '<time class="job-log-time muted">' +
      escapeHtml(fmtMsk(j.created_at)) +
      "</time>" +
      "</a>"
    );
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function qs(obj) {
    var p = new URLSearchParams();
    Object.keys(obj || {}).forEach(function (k) {
      if (obj[k] != null && obj[k] !== "") p.set(k, obj[k]);
    });
    var s = p.toString();
    return s ? "?" + s : "";
  }

  function readUrlFilters() {
    var u = new URL(window.location.href);
    return {
      status: u.searchParams.get("status") || "",
      kind: u.searchParams.get("kind") || "",
      user: u.searchParams.get("user") || "",
      page: parseInt(u.searchParams.get("page") || "1", 10) || 1,
      service: u.searchParams.get("service") || "",
      level: u.searchParams.get("level") || "",
    };
  }

  function writeUrlFilters(f) {
    var u = new URL(window.location.href);
    ["status", "kind", "user", "page", "service", "level"].forEach(function (k) {
      if (f[k]) u.searchParams.set(k, f[k]);
      else u.searchParams.delete(k);
    });
    history.replaceState(null, "", u.pathname + u.search);
  }

  function getCookie(name) {
    var m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
    return m ? decodeURIComponent(m[2]) : "";
  }

  function fetchJson(url, opts) {
    opts = opts || {};
    opts.credentials = "same-origin";
    opts.headers = opts.headers || {};
    if (opts.method && opts.method.toUpperCase() !== "GET") {
      opts.headers["X-CSRF-Token"] = csrfToken || getCookie("csrf_token");
    }
    return fetch(url, opts).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw Object.assign(new Error(j.error || r.statusText), { status: r.status, body: j });
        return j;
      });
    });
  }

  function ensureCsrf() {
    csrfToken = getCookie("csrf_token");
    if (csrfToken) return Promise.resolve(csrfToken);
    return fetchJson("/api/csrf").then(function (j) {
      csrfToken = j.csrf_token || getCookie("csrf_token");
      return csrfToken;
    });
  }

  function tickClock() {
    var el = document.getElementById("clock-msk");
    if (!el) return;
    el.textContent = new Date().toLocaleString("ru-RU", {
      timeZone: MSK,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }) + " MSK";
  }

  function setPauseUi() {
    var btn = document.getElementById("btn-pause");
    if (!btn) return;
    btn.classList.toggle("paused", paused);
    btn.textContent = paused ? "▸" : "||";
    btn.title = paused ? "Продолжить" : "Пауза автообновления";
    btn.setAttribute("aria-label", btn.title);
  }

  function healthRu(h) {
    if (h === "ok") return "ОК";
    if (h === "warn") return "внимание";
    if (h === "error") return "ошибка";
    return h || "—";
  }

  function barLevelLabel(level) {
    if (level === "ok") return "норма";
    if (level === "warn") return "нагрузка";
    if (level === "error") return "критично";
    return "н/д";
  }

  function renderHomeHealth(h) {
    var el = document.getElementById("home-health");
    if (!el) return;
    var cluster = h.cluster || {};
    var bars = h.bars || {};
    var nodes = {};
    (cluster.nodes || []).forEach(function (n) {
      nodes[n.id] = n;
    });
    var ingress = (cluster.diagram && cluster.diagram.ingress) || [];
    var disc = null;
    ingress.forEach(function (n) {
      if (n.id === "discord") disc = n;
    });
    var orch = nodes.orchestrator;
    var workers = nodes.workers;
    var queueVal = bars.queue && bars.queue.value != null ? bars.queue.value : 0;
    var pipeBits = [
      "Discord: " + healthRu(disc && disc.health),
      "Оркестратор: " + healthRu(orch && orch.health),
      "Очередь: " + queueVal,
    ];
    if (workers) {
      pipeBits.push(
        "Воркеры: " +
          (workers.gpu_up != null ? workers.gpu_up + "/" + workers.gpu_expected : workers.detail || "—")
      );
    }
    var hostBits = ["cpu", "mem", "gpu"].map(function (k) {
      var b = bars[k] || {};
      var pct = b.pct != null && !isNaN(Number(b.pct)) ? Math.round(Number(b.pct)) + "%" : "н/д";
      var label = k === "mem" ? "RAM" : k.toUpperCase();
      return label + " " + pct + " (" + barLevelLabel(b.level) + ")";
    });
    var discBits = [
      "Бот: " + healthRu(disc && disc.health),
      "Серверы: " + ((h.kpi && h.kpi.guilds) || 0),
      "Активны 15м: " + ((h.kpi && h.kpi.users_active) || 0),
    ];
    el.innerHTML =
      '<div class="home-health-col"><div class="home-health-title">Пайплайн</div><div class="home-health-body">' +
      escapeHtml(pipeBits.join(" / ")) +
      '</div></div><div class="home-health-col"><div class="home-health-title">Хост</div><div class="home-health-body">' +
      escapeHtml(hostBits.join(" / ")) +
      '</div></div><div class="home-health-col"><div class="home-health-title">Discord</div><div class="home-health-body">' +
      escapeHtml(discBits.join(" / ")) +
      "</div></div>";
  }

  function loadHomeTimeseries() {
    var host = document.getElementById("chart-home-ts");
    if (!host || !window.KizaruCharts || !window.KizaruCharts.timeseriesHome) return Promise.resolve();
    var range = homeTsRange || "24h";
    var metrics = ["jobs_created", "jobs_done", "jobs_error"];
    return Promise.all(
      metrics.map(function (m) {
        return fetchJson("/api/stats/timeseries?metric=" + encodeURIComponent(m) + "&range=" + encodeURIComponent(range));
      })
    ).then(function (parts) {
      var map = {};
      metrics.forEach(function (m, i) {
        map[m] = (parts[i] && parts[i].points) || [];
      });
      window.KizaruCharts.timeseriesHome(host, map);
    });
  }

  function schedule() {
    if (timer) clearInterval(timer);
    timer = setInterval(function () {
      if (!paused) refresh();
      tickClock();
    }, refreshSec * 1000);
  }

  function preserveScroll(fn) {
    var y = window.scrollY;
    return Promise.resolve(fn()).then(function (v) {
      window.scrollTo(0, y);
      return v;
    });
  }

  function queueJobRows(jobs, withStart) {
    var cols = withStart ? 7 : 6;
    return (jobs || [])
      .map(function (j) {
        return (
          "<tr><td><a href=\"/jobs/" +
          j.id +
          "\">" +
          j.id +
          "</a></td><td>" +
          avatarHtml(j.avatar_url) +
          escapeHtml(userLabel(j.user, j.display_name)) +
          "</td><td>" +
          escapeHtml(kindLabel(j.kind)) +
          "</td><td>" +
          escapeHtml(workerLabel(j.worker_kind)) +
          "</td><td>" +
          statusChip(j.status) +
          "</td><td>" +
          escapeHtml(fmtMsk(j.created_at)) +
          "</td>" +
          (withStart ? "<td>" + escapeHtml(fmtMsk(j.started_at)) + "</td>" : "") +
          "</tr>"
        );
      })
      .join("") || '<tr><td colspan="' + cols + '" class="muted">пусто</td></tr>';
  }

  function countByStatus(jobs, status) {
    var n = 0;
    (jobs || []).forEach(function (j) {
      if (j.status === status) n += 1;
    });
    return n;
  }

  function setPipeValue(id, value) {
    var el = document.getElementById(id);
    if (!el) return;
    var v = el.querySelector(".v");
    if (v) v.textContent = String(value);
  }

  function renderClusterSchema(el, cluster) {
    if (!el) return;
    if (window.KizaruCharts && typeof window.KizaruCharts.clusterMap === "function") {
      window.KizaruCharts.clusterMap(el, cluster || {});
      return;
    }
    el.innerHTML = '<p class="muted">нет схемы</p>';
  }

  function renderGauges(el, bars) {
    if (!el) return;
    if (window.KizaruCharts && window.KizaruCharts.gauges) {
      window.KizaruCharts.gauges(el, bars || {});
    } else {
      el.innerHTML = '<p class="muted">нет графиков</p>';
    }
  }

  function renderStorageEl(el, storage) {
    if (!el) return;
    if (window.KizaruCharts && window.KizaruCharts.storageBars) {
      window.KizaruCharts.storageBars(el, storage || {});
    } else {
      el.innerHTML = ((storage && storage.items) || [])
        .map(function (it) {
          return (
            "<div>" +
            escapeHtml(it.label) +
            ': <span class="num">' +
            escapeHtml(it.human || it.bytes) +
            "</span></div>"
          );
        })
        .join("");
    }
  }

  function clusterDbInfo(cluster) {
    var out = { file: "", journal: "", queueHealth: "" };
    var bus = cluster && cluster.diagram && cluster.diagram.bus;
    if (bus) {
      bus.forEach(function (c) {
        if (c.id === "queue") out.file = c.detail || "";
        if (c.id === "sqlite") out.journal = c.detail || "";
      });
    }
    (cluster && cluster.nodes || []).forEach(function (n) {
      if (n.id === "queue") out.queueHealth = n.health || "";
    });
    return out;
  }

  function renderHome() {
    return fetchJson("/api/home").then(function (h) {
      var stamp = document.getElementById("home-stamp");
      if (stamp) stamp.textContent = "Обновлено: " + fmtMsk(h.generated_at);
      renderHomeHealth(h);
      var cluster = document.getElementById("cluster");
      if (cluster) {
        renderClusterSchema(cluster, h.cluster || {});
      }
      var kpi = document.getElementById("home-kpi");
      var k = h.kpi || {};
      if (kpi) {
        var items = [
          ["Активные", k.active],
          ["Готово 24ч", k.done_24h],
          ["Ошибки 24ч", k.error_24h],
          ["Активны 15м", k.users_active],
          ["Серверы", k.guilds],
        ];
        kpi.innerHTML = items
          .map(function (it) {
            return (
              '<div class="kpi"><div class="v">' +
              escapeHtml(it[1]) +
              '</div><div class="l">' +
              escapeHtml(it[0]) +
              "</div></div>"
            );
          })
          .join("");
      }
      if (window.KizaruCharts && window.KizaruCharts.statusDonut) {
        window.KizaruCharts.statusDonut(
          document.getElementById("chart-home-status"),
          h.by_status || {}
        );
      }
      renderGauges(document.getElementById("home-bars"), h.bars);
      if (window.KizaruCharts && window.KizaruCharts.heatmap) {
        window.KizaruCharts.heatmap(document.getElementById("chart-home-heatmap"), h.heatmap || {});
      }
      if (window.KizaruCharts && window.KizaruCharts.audience) {
        window.KizaruCharts.audience(
          document.getElementById("chart-home-audience"),
          h.audience || {},
          h.bot_stats || {}
        );
      }
      renderStorageEl(document.getElementById("home-storage"), h.storage);
      var jobLog = document.getElementById("tbl-home-jobs");
      if (jobLog) {
        var recent = h.recent_jobs || [];
        jobLog.innerHTML = recent.length
          ? recent.map(homeJobLogRow).join("")
          : '<p class="job-log-empty muted">Нет недавних задач. <a href="/jobs">Открыть задачи</a></p>';
      }
      return loadHomeTimeseries();
    });
  }

  function renderSystem() {
    return fetchJson("/api/home").then(function (h) {
      var stamp = document.getElementById("system-stamp");
      if (stamp) stamp.textContent = "Обновлено: " + fmtMsk(h.generated_at);
      var meta = document.getElementById("system-db-meta");
      if (meta) {
        var db = clusterDbInfo(h.cluster || {});
        var rows = [
          ["Доступность", h.db_ok ? "файл на месте" : "нет файла"],
          ["Файл", db.file || "—"],
          ["Журнал", db.journal || "—"],
          ["Очередь", db.queueHealth || "—"],
        ];
        meta.innerHTML =
          '<table class="attrs">' +
          rows
            .map(function (r) {
              return "<tr><th>" + escapeHtml(r[0]) + "</th><td>" + escapeHtml(r[1]) + "</td></tr>";
            })
            .join("") +
          "</table>";
      }
      renderGauges(document.getElementById("system-bars"), h.bars);
      renderStorageEl(document.getElementById("system-storage"), h.storage);
    });
  }

  function renderQueue() {
    return fetchJson("/api/queue").then(function (q) {
      var jobs = q.jobs || [];
      var pipe = q.pipeline || {};
      var queued = pipe.queued != null ? pipe.queued : countByStatus(jobs, "queued");
      var assigned = pipe.assigned != null ? pipe.assigned : countByStatus(jobs, "assigned");
      var running = pipe.running != null ? pipe.running : countByStatus(jobs, "running");
      var clients = pipe.clients != null ? pipe.clients : jobs.length;
      setPipeValue("pipe-clients", clients);
      setPipeValue("pipe-bus", queued);
      setPipeValue("pipe-orch", assigned);
      setPipeValue("pipe-workers", running);
      var tb = document.querySelector("#tbl-queue-page tbody");
      if (tb) {
        tb.innerHTML = jobs.length
          ? queueJobRows(jobs, true)
          : '<tr><td colspan="7" class="muted">Очередь пуста. <a href="/jobs">История задач</a></td></tr>';
      }
    });
  }

  function renderJobs() {
    var parts = [];
    if (document.getElementById("queue-pipeline")) {
      parts.push(renderQueue());
    }
    var f = readUrlFilters();
    var form = document.getElementById("jobs-filters");
    if (form) {
      form.status.value = f.status;
      form.kind.value = f.kind;
      form.user.value = f.user;
    }
    document.querySelectorAll("#status-chips .chip").forEach(function (chip) {
      var st = chip.getAttribute("data-status") || "";
      if (st === f.status) chip.classList.add("active");
      else chip.classList.remove("active");
    });
    parts.push(
      fetchJson(
        "/api/jobs" +
          qs({ status: f.status, kind: f.kind, user: f.user, page: f.page })
      ).then(function (data) {
      var tb = document.querySelector("#tbl-jobs tbody");
      if (tb) {
        tb.innerHTML = (data.jobs || [])
          .map(function (j) {
            return (
              "<tr><td><a href=\"/jobs/" +
              j.id +
              "\">" +
              j.id +
              "</a></td><td>" +
              avatarHtml(j.avatar_url) +
              "<a href=\"/users/" +
              encodeURIComponent(j.user) +
              "\">" +
              escapeHtml(userLabel(j.user, j.display_name)) +
              "</a></td><td>" +
              escapeHtml(kindLabel(j.kind)) +
              "</td><td>" +
              escapeHtml(workerLabel(j.worker_kind)) +
              "</td><td>" +
              statusChip(j.status) +
              "</td><td>" +
              escapeHtml(fmtMsk(j.created_at)) +
              "</td><td class=\"mono\">" +
              escapeHtml(j.preview || "") +
              "</td></tr>"
            );
          })
          .join("") || '<tr><td colspan="7" class="muted">пусто</td></tr>';
      }
      var pager = document.getElementById("jobs-pager");
      if (pager) {
        var pages = Math.max(1, Math.ceil((data.total || 0) / (data.page_size || 50)));
        pager.innerHTML =
          '<button type="button" data-p="' +
          Math.max(1, f.page - 1) +
          '">‹</button> <span id="jobs-page-num">' +
          f.page +
          "</span> / " +
          pages +
          ' <button type="button" data-p="' +
          Math.min(pages, f.page + 1) +
          '">›</button> <span class="muted">всего ' +
          (data.total || 0) +
          "</span>";
        pager.querySelectorAll("button").forEach(function (b) {
          b.onclick = function () {
            f.page = parseInt(b.getAttribute("data-p"), 10) || 1;
            writeUrlFilters(f);
            refresh();
          };
        });
      }
      if (window.KizaruCharts && window.KizaruCharts.timelineGantt) {
        window.KizaruCharts.timelineGantt(
          document.getElementById("chart-jobs-gantt"),
          (data.jobs || []).map(function (j) {
            return {
              id: j.id,
              start: j.created_at,
              end: j.finished_at || j.started_at || j.created_at,
              status: j.status,
              label: "#" + j.id,
            };
          })
        );
      }
      })
    );
    return Promise.all(parts);
  }

  function renderJobDetail() {
    var box = document.getElementById("job-detail");
    if (!box) return Promise.resolve();
    var id = box.getAttribute("data-job-id");
    return fetchJson("/api/jobs/" + id).then(function (j) {
      var cancelBtn = j.cancellable
        ? '<p><button type="button" id="btn-cancel">Отменить</button></p>'
        : "";
      box.innerHTML =
        "<table class=\"attrs\">" +
        [
          ["Статус", statusChip(j.status)],
          ["Пользователь", '<a href="/users/' + encodeURIComponent(j.user) + '">' + escapeHtml(userLabel(j.user, j.display_name)) + "</a>"],
          ["kind", escapeHtml(kindLabel(j.kind))],
          ["worker_kind", escapeHtml(workerLabel(j.worker_kind))],
          ["created", escapeHtml(fmtMsk(j.created_at))],
          ["started", escapeHtml(fmtMsk(j.started_at))],
          ["finished", escapeHtml(fmtMsk(j.finished_at))],
          ["pid", escapeHtml(j.worker_pid || "—")],
          ["attempts", escapeHtml(j.attempts)],
          ["payload", '<pre class="mono">' + escapeHtml(j.payload_preview || "") + "</pre>"],
          ["result", '<pre class="mono">' + escapeHtml(j.result_preview || "") + "</pre>"],
          ["error", '<pre class="mono bad">' + escapeHtml(j.error || "") + "</pre>"],
        ]
          .map(function (r) {
            return "<tr><th>" + r[0] + "</th><td>" + r[1] + "</td></tr>";
          })
          .join("") +
        "</table>" +
        cancelBtn;
      var btn = document.getElementById("btn-cancel");
      if (btn) {
        btn.onclick = function () {
          ensureCsrf().then(function () {
            return fetchJson("/api/jobs/" + id + "/cancel", { method: "POST" });
          }).then(function () {
            refresh();
          }).catch(function (e) {
            alert((e.body && e.body.error) || e.message);
          });
        };
      }
    });
  }

  function renderActivityBars(el, buckets) {
    if (!el) return;
    buckets = buckets || [];
    if (!buckets.length) {
      el.innerHTML =
        '<div class="empty">' + (el.getAttribute("data-empty") || "нет данных") + "</div>";
      return;
    }
    var max = 0;
    buckets.forEach(function (b) {
      max = Math.max(max, Number(b.count) || 0);
    });
    if (!max) max = 1;
    el.innerHTML = buckets
      .map(function (b) {
        var n = Number(b.count) || 0;
        var pct = Math.round((100 * n) / max);
        return (
          '<div class="activity-row">' +
          '<span class="label">' +
          escapeHtml(b.label || "") +
          "</span>" +
          '<div class="activity-track"><div class="activity-fill" style="width:' +
          pct +
          '%"></div></div>' +
          '<span class="num">' +
          n +
          "</span></div>"
        );
      })
      .join("");
  }

  function renderUsers() {
    var hide = document.getElementById("hide-cli");
    var hideCli = hide ? hide.checked : true;
    return fetchJson("/api/users" + (hideCli ? "?hide_cli=1" : "")).then(function (data) {
      var stamp = document.getElementById("users-stamp");
      if (stamp) stamp.textContent = "Обновлено: " + fmtMsk(data.generated_at);
      var s = data.summary || {};
      var kpi = document.getElementById("users-kpi");
      if (kpi) {
        var items = [
          ["Записей в БД", s.users_total],
          ["Серверы", s.guilds],
          ["Активны 15м", s.users_active_15m],
          ["Активны 24ч", s.users_active_24h],
          ["С задачами", s.users_with_jobs],
          ["Задач в работе", s.jobs_active],
        ];
        if (!hideCli && s.cli_rows) {
          items.push(["cli записей", s.cli_rows]);
        }
        var bs = (data.bot_stats && data.bot_stats.latest) || {};
        if (bs.members_total != null) {
          items.push(["Участники Discord", bs.members_total]);
        }
        if (s.guild_members_sum) {
          items.push(["Σ member_count", s.guild_members_sum]);
        }
        kpi.innerHTML = items
          .map(function (it) {
            return (
              '<div class="kpi"><div class="v">' +
              escapeHtml(it[1]) +
              '</div><div class="l">' +
              escapeHtml(it[0]) +
              "</div></div>"
            );
          })
          .join("");
      }
      if (window.KizaruCharts && window.KizaruCharts.audience) {
        window.KizaruCharts.audience(
          document.getElementById("chart-users-audience"),
          data.audience || {},
          data.bot_stats || {}
        );
      }
      if (window.KizaruCharts && window.KizaruCharts.sparkline) {
        window.KizaruCharts.sparkline(
          document.getElementById("chart-users-jobs-spark"),
          data.jobs_sparkline_7d || []
        );
      }
      var cap = document.getElementById("users-jobs-caption");
      if (cap) {
        var kj = data.kpi_jobs || {};
        cap.textContent =
          "Создано задач по дням (7д). Готово 24ч: " +
          (kj.done_24h != null ? kj.done_24h : "—") +
          ", ошибки 24ч: " +
          (kj.error_24h != null ? kj.error_24h : "—");
      }
      var act = data.activity || {};
      renderActivityBars(document.getElementById("users-activity-recency"), act.by_recency);
      renderActivityBars(document.getElementById("users-activity-jobs"), act.by_jobs);
      var tb = document.querySelector("#tbl-users tbody");
      if (!tb) return;
      tb.innerHTML = (data.users || [])
        .map(function (u) {
          var href = u.key != null ? "/users/" + encodeURIComponent(u.key) : "#";
          return (
            "<tr><td>" +
            '<a href="' +
            href +
            '">#' +
            escapeHtml(String(u.key)) +
            "</a></td><td class=\"num\">" +
            u.total +
            "</td><td class=\"num\">" +
            u.active +
            "</td><td>" +
            escapeHtml(fmtMsk(u.last_seen)) +
            "</td></tr>"
          );
        })
        .join("") || '<tr><td colspan="4" class="muted">пусто</td></tr>';
    });
  }

  function renderUserDetail() {
    var box = document.getElementById("user-detail");
    if (!box) return Promise.resolve();
    var id = box.getAttribute("data-user-id");
    return fetchJson("/api/users/" + encodeURIComponent(id)).then(function (u) {
      var base =
        "<table class=\"attrs\">" +
        [
          ["id", escapeHtml(u.id_mask || u.id)],
          ["имя", "—"],
          ["всего", u.total],
          ["ожидание ср.", fmtDur(u.mean_wait_sec)],
          ["выполнение ср.", fmtDur(u.mean_run_sec)],
          ["first_seen", fmtMsk(u.first_seen)],
          ["last_seen", fmtMsk(u.last_seen)],
        ]
          .map(function (r) {
            return "<tr><th>" + r[0] + "</th><td>" + r[1] + "</td></tr>";
          })
          .join("") +
        "</table>" +
        '<p><button type="button" id="btn-reveal-profile">Показать профиль</button> ' +
        '<button type="button" id="btn-delete-user">Удалить</button></p>' +
        '<div id="user-pii" class="muted"></div>';
      box.innerHTML = base;
      var spark = Object.keys(u.by_status || {}).map(function (k) {
        return { t: k, v: u.by_status[k] };
      });
      if (window.KizaruCharts && window.KizaruCharts.sparkline) {
        window.KizaruCharts.sparkline(document.getElementById("chart-user-spark"), spark);
      }
      var reveal = document.getElementById("btn-reveal-profile");
      if (reveal) {
        reveal.addEventListener("click", function () {
          ensureCsrf()
            .then(function () {
              return fetchJson("/api/users/" + encodeURIComponent(id) + "/profile", {
                method: "POST",
              });
            })
            .then(function (p) {
              var pii = document.getElementById("user-pii");
              if (!pii) return;
              pii.innerHTML =
                avatarHtml(p.avatar_url, "avatar-lg") +
                " id=" +
                escapeHtml(p.id) +
                " — " +
                escapeHtml(p.display_name || "—");
            })
            .catch(function (e) {
              console.error(e);
            });
        });
      }
      var del = document.getElementById("btn-delete-user");
      if (del) {
        del.addEventListener("click", function () {
          if (!window.confirm("Удалить пользователя и задачи?")) return;
          ensureCsrf()
            .then(function () {
              return fetchJson("/api/users/" + encodeURIComponent(id) + "/delete", {
                method: "POST",
              });
            })
            .then(function () {
              window.location.href = "/users";
            })
            .catch(function (e) {
              console.error(e);
            });
        });
      }
    });
  }

  var WORKER_PROD_CKPT = {
    text_lstm: "kizaru.pt",
    text_lstm_word: "kizaru_word.pt",
    text_transformer: "transformer.pt",
    text_gpt: "gpt_ru_rap_v1-prod.pt",
    text_deepseek: "deepseek.pt",
    text_bert_gen: "bert_gen.pt",
    mood: "mood.pt",
  };

  function modelHealthClass(m) {
    if (!m) return "meta";
    if (!m.present) return "offline";
    if (m.missing_metadata) return "meta";
    return "on";
  }

  function workerDotClass(w) {
    if (!w || !w.running) return "offline";
    return "on";
  }

  function renderServices() {
    return fetchJson("/api/services").then(function (data) {
      var stamp = document.getElementById("services-stamp");
      if (stamp) stamp.textContent = "Обновлено: " + fmtMsk(data.generated_at);
      var note = document.getElementById("services-ops-note");
      if (note) note.textContent = data.ops_note || "";
      var sum = document.getElementById("services-summary");
      if (sum) {
        sum.textContent =
          "процессов: " +
          (data.running_count || 0) +
          " / " +
          (data.expected_count || 0) +
          " (ожидаемых модулей)";
      }
      var deploy = document.getElementById("services-deploy-cmd");
      if (deploy) deploy.textContent = data.deploy_all || "";
      var pidFile = document.getElementById("services-pid-file");
      if (pidFile && data.pid_file) pidFile.textContent = data.pid_file;
      var tb = document.querySelector("#tbl-services tbody");
      if (tb) {
        tb.innerHTML = (data.services || [])
          .map(function (s) {
            var logHref = s.log_stem ? "/logs?service=" + encodeURIComponent(s.log_stem) : "/logs";
            return (
              "<tr><td><span class=\"dot " +
              escapeHtml(s.health || (s.running ? "on" : "off")) +
              "\"></span></td><td>" +
              escapeHtml(s.label || s.key) +
              "</td><td class=\"mono\">" +
              escapeHtml(s.module || "") +
              "</td><td class=\"mono\">" +
              escapeHtml(s.pids || "—") +
              "</td><td class=\"num\">" +
              escapeHtml(s.rss_human || "—") +
              "</td><td>" +
              (s.db_meta ? "да" : "нет") +
              '</td><td><a href="' +
              logHref +
              '">лог</a></td><td class="mono">' +
              escapeHtml(s.manual_cmd || "—") +
              "</td></tr>"
            );
          })
          .join("");
      }
    });
  }

  function renderWorkers() {
    return Promise.all([fetchJson("/api/workers"), fetchJson("/api/models")]).then(function (parts) {
      var data = parts[0];
      var mdata = parts[1];
      var byName = {};
      (mdata.models || []).forEach(function (m) {
        byName[m.name] = m;
      });
      var pairedNames = {};
      var sum = document.getElementById("workers-summary");
      if (sum) {
        sum.textContent =
          "orchestrator: " +
          (data.orchestrator && data.orchestrator.running ? "●" : "—") +
          "  discord: " +
          (data.discord && data.discord.running ? "●" : "—");
      }
      var tb = document.querySelector("#tbl-runtime tbody");
      if (tb) {
        tb.innerHTML = (data.workers || [])
          .map(function (w) {
            var ckpt = WORKER_PROD_CKPT[w.worker_kind] || "";
            var m = ckpt ? byName[ckpt] : null;
            if (ckpt) pairedNames[ckpt] = true;
            return (
              "<tr><td><span class=\"dot " +
              workerDotClass(w) +
              "\"></span></td><td class=\"mono\">" +
              escapeHtml(w.key) +
              "</td><td>" +
              escapeHtml(workerLabel(w.worker_kind)) +
              "</td><td class=\"mono\">" +
              escapeHtml(ckpt || "—") +
              "</td><td><span class=\"dot " +
              modelHealthClass(m) +
              "\"></span></td><td class=\"num\">" +
              escapeHtml(m && m.step != null ? m.step : "—") +
              "</td><td class=\"num\">" +
              escapeHtml(m && m.val_ru_rap != null ? m.val_ru_rap : "—") +
              "</td><td class=\"mono\">" +
              escapeHtml(w.pids || "—") +
              "</td><td class=\"num\">" +
              (w.rss || 0) +
              "</td><td>" +
              (w.db_meta ? "да" : "нет") +
              "</td></tr>"
            );
          })
          .join("");
      }
      var extra = document.querySelector("#tbl-models-extra tbody");
      if (extra) {
        var orphans = (mdata.models || []).filter(function (m) {
          return !pairedNames[m.name];
        });
        extra.innerHTML =
          orphans
            .map(function (m) {
              if (m.missing_metadata) {
                return (
                  "<tr><td class=\"mono\">" +
                  escapeHtml(m.name) +
                  '</td><td colspan="5" class="muted">нет метаданных</td></tr>'
                );
              }
              return (
                "<tr><td class=\"mono\">" +
                escapeHtml(m.name) +
                "</td><td>" +
                escapeHtml(m.tokenizer || "—") +
                "</td><td class=\"num\">" +
                escapeHtml(m.step != null ? m.step : "—") +
                "</td><td class=\"num\">" +
                escapeHtml(m.val_ru_rap != null ? m.val_ru_rap : "—") +
                "</td><td class=\"num\">" +
                escapeHtml(m.bpc != null ? m.bpc : "—") +
                "</td><td>" +
                escapeHtml(m.date ? fmtMsk(m.date) : "—") +
                "</td></tr>"
              );
            })
            .join("") || '<tr><td colspan="6" class="muted">нет</td></tr>';
      }
      if (window.KizaruCharts && window.KizaruCharts.fleetMatrix) {
        window.KizaruCharts.fleetMatrix(
          document.getElementById("chart-workers-matrix"),
          data.workers || [],
          mdata.models || [],
          WORKER_PROD_CKPT
        );
      }
      var loss = (mdata.models || [])
        .filter(function (m) {
          return m.val_ru_rap != null;
        })
        .map(function (m) {
          return { t: m.name, v: Number(m.val_ru_rap) };
        });
      if (window.KizaruCharts && window.KizaruCharts.lineChart) {
        window.KizaruCharts.lineChart(document.getElementById("chart-models-loss"), loss, { categorical: true });
      }
    });
  }

  var lastLogLines = [];

  function logLevelKey(level) {
    var u = String(level || "INFO").toUpperCase();
    if (u === "WARN" || u === "WARNING") return "warning";
    if (u === "DEBUG") return "debug";
    if (u === "ERROR" || u === "CRITICAL" || u === "FATAL") return "error";
    return "info";
  }

  function logLevelLabel(level) {
    var u = String(level || "INFO").toUpperCase();
    if (u === "WARN") return "WARNING";
    return u;
  }

  function escapeRegExp(s) {
    return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function highlightSearch(text, query) {
    var esc = escapeHtml(text);
    var q = (query || "").trim();
    if (!q) return esc;
    try {
      var re = new RegExp(escapeRegExp(q), "gi");
      return esc.replace(re, function (m) {
        return '<mark class="log-hit">' + m + "</mark>";
      });
    } catch (e) {
      return esc;
    }
  }

  function logLineHaystack(line) {
    return [line.t, line.level, line.service, line.msg]
      .map(function (v) {
        return v == null ? "" : String(v);
      })
      .join("\n")
      .toLowerCase();
  }

  function syncLogWrapUi() {
    var view = document.getElementById("log-view");
    var box = document.getElementById("logs-wrap");
    if (!view || !box) return;
    if (box.checked) view.classList.add("log-wrap");
    else view.classList.remove("log-wrap");
  }

  function paintLogLines() {
    var el = document.getElementById("log-lines");
    var meta = document.getElementById("logs-meta");
    if (!el) return;
    syncLogWrapUi();
    var searchEl = document.getElementById("logs-search");
    var query = searchEl ? searchEl.value : "";
    var qLower = query.trim().toLowerCase();
    var lines = lastLogLines || [];
    var shown = lines;
    if (qLower) {
      shown = lines.filter(function (l) {
        return logLineHaystack(l).indexOf(qLower) !== -1;
      });
    }
    if (meta) {
      if (!lines.length) {
        meta.textContent = "нет строк с сервера";
      } else if (qLower && shown.length !== lines.length) {
        meta.textContent = "показано " + shown.length + " из " + lines.length;
      } else {
        meta.textContent = "строк: " + lines.length;
      }
    }
    if (!shown.length) {
      el.innerHTML = '<div class="log-empty">' + escapeHtml(qLower ? "нет совпадений" : "нет данных") + "</div>";
      return;
    }
    el.innerHTML = shown
      .map(function (l) {
        var lvlKey = logLevelKey(l.level);
        var ts = l.t ? fmtMsk(l.t) : "";
        return (
          '<div class="log-row log-level-' +
          lvlKey +
          '">' +
          '<span class="log-ts">' +
          escapeHtml(ts) +
          "</span>" +
          '<span class="log-lvl log-lvl-' +
          lvlKey +
          '">' +
          escapeHtml(logLevelLabel(l.level)) +
          "</span>" +
          '<span class="log-svc" title="' +
          escapeHtml(l.service || "") +
          '">' +
          escapeHtml(l.service || "—") +
          "</span>" +
          '<span class="log-msg">' +
          highlightSearch(l.msg || "", query) +
          "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderLogs() {
    var f = readUrlFilters();
    var form = document.getElementById("logs-filters");
    if (form) {
      if (form.service) form.service.value = f.service;
      if (form.level) form.level.value = f.level || "";
    }
    return fetchJson("/api/logs" + qs({ service: f.service, level: f.level })).then(function (data) {
      lastLogLines = data.lines || [];
      paintLogLines();
    });
  }

  function refresh() {
    var fn =
      {
        home: renderHome,
        services: renderServices,
        system: renderSystem,
        jobs: page === "jobs" && document.getElementById("job-detail") ? renderJobDetail : renderJobs,
        users: document.getElementById("user-detail") ? renderUserDetail : renderUsers,
        workers: renderWorkers,
        logs: renderLogs,
      }[page] || renderHome;
    if (document.getElementById("job-detail")) fn = renderJobDetail;
    if (document.getElementById("user-detail")) fn = renderUserDetail;
    return preserveScroll(fn).catch(function (e) {
      console.error(e);
    });
  }

  var jobsForm = document.getElementById("jobs-filters");
  if (jobsForm) {
    jobsForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var f = readUrlFilters();
      f.status = jobsForm.status.value.trim();
      f.kind = jobsForm.kind.value.trim();
      f.user = jobsForm.user.value.trim();
      f.page = 1;
      writeUrlFilters(f);
      refresh();
    });
  }
  var logsForm = document.getElementById("logs-filters");
  if (logsForm) {
    logsForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var f = readUrlFilters();
      f.service = logsForm.service.value.trim();
      f.level = logsForm.level.value.trim();
      writeUrlFilters(f);
      refresh();
    });
  }
  var logsSearch = document.getElementById("logs-search");
  if (logsSearch) {
    logsSearch.addEventListener("input", function () {
      paintLogLines();
    });
  }
  var logsWrap = document.getElementById("logs-wrap");
  if (logsWrap) {
    logsWrap.addEventListener("change", function () {
      paintLogLines();
    });
  }
  var hideCli = document.getElementById("hide-cli");
  if (hideCli) hideCli.addEventListener("change", refresh);
  var pauseBtn = document.getElementById("btn-pause");
  if (pauseBtn) {
    pauseBtn.addEventListener("click", function () {
      paused = !paused;
      localStorage.setItem(PAUSE_KEY, paused ? "1" : "0");
      setPauseUi();
    });
  }
  var tsRange = document.getElementById("home-ts-range");
  if (tsRange) {
    tsRange.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button[data-range]");
      if (!btn) return;
      homeTsRange = btn.getAttribute("data-range") || "24h";
      tsRange.querySelectorAll("button[data-range]").forEach(function (b) {
        var on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      loadHomeTimeseries().catch(function (e) {
        console.error(e);
      });
    });
  }
  var navToggle = document.getElementById("btn-nav-toggle");
  var mainNav = document.getElementById("main-nav");
  if (navToggle && mainNav) {
    navToggle.addEventListener("click", function () {
      var open = mainNav.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  setPauseUi();
  tickClock();
  ensureCsrf().then(refresh);
  schedule();
})();
