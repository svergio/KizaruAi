(function (global) {
  "use strict";
  global.KizaruCharts = global.KizaruCharts || {};

  var FILL_CLASS = ["storage-fill-a", "storage-fill-b", "storage-fill-c"];

  function pctOfPart(bytes, total) {
    if (!total || total <= 0) return 0;
    return Math.max(0, Math.min(100, (bytes / total) * 100));
  }

  function pctLabel(bytes, total) {
    if (!total || total <= 0) return "—";
    return Math.round(pctOfPart(bytes, total)) + "%";
  }

  global.KizaruCharts.storageBars = function (el, payload) {
    if (!el) return;
    el.innerHTML = "";
    var items = (payload && payload.items) || [];
    if (!items.length) {
      var empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = el.getAttribute("data-empty") || "нет данных";
      el.appendChild(empty);
      return;
    }

    var total = 0;
    var maxBytes = 0;
    items.forEach(function (it) {
      var b = Number(it.bytes) || 0;
      total += b;
      if (b > maxBytes) maxBytes = b;
    });
    var scale = maxBytes > 0 ? maxBytes : 1;

    var stack = document.createElement("div");
    stack.className = "storage-stack";
    stack.setAttribute("role", "img");
    stack.setAttribute(
      "aria-label",
      items
        .map(function (it) {
          return (it.label || "") + " " + (it.human || it.bytes);
        })
        .join(", ")
    );

    items.forEach(function (it, i) {
      var b = Number(it.bytes) || 0;
      if (total > 0 && b > 0) {
        var seg = document.createElement("div");
        seg.className = "storage-stack-seg " + (FILL_CLASS[i % FILL_CLASS.length] || "");
        seg.style.width = pctOfPart(b, total) + "%";
        stack.appendChild(seg);
      }
    });
    if (stack.childNodes.length) {
      el.appendChild(stack);
    }

    items.forEach(function (it, i) {
      var fillCls = FILL_CLASS[i % FILL_CLASS.length] || "";
      var bytes = Number(it.bytes) || 0;

      var row = document.createElement("div");
      row.className = "bar-row storage-bar-row";

      var labelCell = document.createElement("div");
      labelCell.className = "storage-label";
      var labelMain = document.createElement("div");
      labelMain.className = "storage-label-main";
      labelMain.textContent = it.label || "";
      var labelPath = document.createElement("div");
      labelPath.className = "storage-label-path muted";
      labelPath.textContent = it.path || "";
      labelCell.appendChild(labelMain);
      labelCell.appendChild(labelPath);

      var track = document.createElement("div");
      track.className = "bar-track";
      var fill = document.createElement("div");
      fill.className = "bar-fill " + fillCls;
      fill.style.width = (bytes / scale) * 100 + "%";
      track.appendChild(fill);

      var pctCell = document.createElement("div");
      pctCell.className = "storage-pct num muted";
      pctCell.textContent = pctLabel(bytes, total);

      var sizeCell = document.createElement("div");
      sizeCell.className = "storage-size num";
      sizeCell.textContent = it.human != null && it.human !== "" ? String(it.human) : String(bytes);

      row.appendChild(labelCell);
      row.appendChild(track);
      row.appendChild(pctCell);
      row.appendChild(sizeCell);
      el.appendChild(row);
    });
  };
})(window);
