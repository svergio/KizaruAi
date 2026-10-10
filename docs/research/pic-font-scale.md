# Pic: шрифт × scale (без upscale)

Источник: картинка [PluviumG / X](https://x.com/PluviumG/status/2107559081116660149) (локальная копия и выгрузки удалены 2026-10-09; выводы ниже сохранены).

| knobs | смысл |
|-------|--------|
| `scale` | px на квадрант с **оригинала**; смайл = `2×scale` |
| `font` / `smile_px` | размер тайла глифа в **выходном** PNG |
| `upscale=False` | cols = `src_w // (2·scale)` — картинка **сжимается** при росте scale |

CLI: `--font N --no-upscale --scale S`

```bash
python notebooks/research/pic_font_scale_stills.py
```

- **A:** `scale=8`, fonts `4 6 8 12 16 24` × line/render/details  
- **B:** пары `font = 2·scale` → `(4,8) (6,12) (8,16) (12,24) (14,28) (16,32)` × 3 modes  
  (блок с оригинала = размер тайла глифа, смотрим ужатие сетки)

Артефакты прогона удалены 2026-10-09; при повторном запуске `notebooks/research/pic_font_scale_stills.py` их можно пересоздать.

## Render @ huge scale

При `structure=color` и `scale≥24` маска падает на **luma** (tone остаётся OKLab), иначе color-edges дают шум/полосы. На `scale` 5…23 color-mask только грубеет (`_adapt_mask_opts`).
