import argparse
import asyncio
import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from core.bus.queue import (
    DEFAULT_POLL_TIMEOUT,
    MAX_ATTACHMENT_BYTES,
    ActiveJobError,
    JobStore,
    PayloadTooLargeError,
    QueueFullError,
    format_eta_ru,
    poll_until_done,
)
from core.bus.user_stats import format_me_embed_lines
from core.mood.model import mood_source
from core.bus.routing import build_pic_payload, build_text_payload, resolve_worker_kind
from core.image.ascii import (
    CLI_DEFAULT_CELLS,
    CLI_DEFAULT_SCALE,
    CLI_MAX_COLS,
)
from core.text.params import (
    CHAR_LENGTH_MAX,
    CHAR_LENGTH_MIN,
    ITERS_MAX,
    ITERS_MIN,
    LINES_MAX,
    LINES_MIN,
    TEMP_MAX,
    TEMP_MIN,
    WORD_LENGTH_MAX,
    WORD_LENGTH_MIN,
)

DISCORD_LIMIT = 1800
BOT_STATS_INTERVAL_SEC = 60.0
_STILL_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_ANIM_EXT = (".gif", ".mp4", ".webm", ".mov")

ACTIVE_JOB_MSG = "У тебя уже есть заявка в очереди. Дождись результата."
GENERIC_ERROR_MSG = "Не удалось выполнить запрос. Попробуй позже."
ATTACHMENT_TOO_LARGE_MSG = (
    f"Вложение слишком большое (лимит {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB)."
)
PAYLOAD_TOO_LARGE_MSG = "Запрос слишком большой для очереди."
QUEUE_FULL_MSG = "Очередь переполнена. Попробуй через несколько минут."
SERVICE_OFFLINE_MSG = (
    "Модуль {name} сейчас не активен. "
    "Попробуй позже или выбери другой."
)
DEFAULT_DB = Path("data/bot.db")

log = logging.getLogger("kizaru.bot")

AI_HELP = (
    "**/ai text** — продолжение текста. "
    "`model`: lstm (символы), lstm-word (слова), transformer, gpt, deepseek или bert-gen. "
    "`start` по умолчанию «Я». "
    "Если модуль модели не активен, бот ответит сразу.\n"
    f"Параметры генерации необязательны, пусто — дефолт модели: "
    f"`temp` {TEMP_MIN}-{TEMP_MAX} (ниже — повторы строк, выше — битые слова), "
    f"`lines` {LINES_MIN}-{LINES_MAX} строк, "
    f"`length` бюджет в символах ({CHAR_LENGTH_MIN}-{CHAR_LENGTH_MAX}), "
    f"у lstm-word в словах ({WORD_LENGTH_MIN}-{WORD_LENGTH_MAX}), "
    f"`iters` только у bert-gen.\n"
    "**/ai mood** — две оси: добрый/злой и веселый/грустный. "
    "`text` необязателен: пустой берёт reply, иначе вставь текст.\n"
    "**/ai pic** — символьная картинка, нужно вложение. "
    "`mode`: **simple** | **line** | **render** | **details**. "
    "По умолчанию simple/line/details — текст в чат (31 колонка); "
    "render — PNG/GIF. "
    "`cli=True` — ответ файлом (как CLI still), для любого режима. "
    "`scale`/`cells` для файла (по умолчанию 2 и 320); "
    "`dark`/`light` только simple; `line_thr`/`details_thr` у line/render/details.\n"
    "**/ai cancel** — отменить свою активную заявку (queued, assigned или running).\n"
    "**/ai me** — статистика твоих заявок.\n"
    "**/ai help** — эта справка."
)


async def _reference_text(channel, reference):
    if reference is None or reference.message_id is None:
        return ""
    message = reference.resolved
    if message is None:
        try:
            message = await channel.fetch_message(reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            return ""
    return message.content or ""


async def replied_content(ctx):
    reference = ctx.message.reference
    if reference is None or reference.message_id is None:
        return ""
    message = reference.resolved
    if message is None:
        channel = getattr(ctx, "channel", None)
        if channel is None:
            return ""
        return await _reference_text(channel, reference)
    return message.content or ""


def _is_media_name(name, content_type):
    name = (name or "").lower()
    content_type = content_type or ""
    if content_type.startswith("image/") or content_type.startswith("video/"):
        return True
    return name.endswith(_STILL_EXT + _ANIM_EXT)


async def _attachment_bytes(attachment):
    if attachment is None:
        return None, None
    if not _is_media_name(attachment.filename, attachment.content_type):
        return None, None
    size = getattr(attachment, "size", None)
    if size is not None and size > MAX_ATTACHMENT_BYTES:
        return None, "too_large"
    data = await attachment.read()
    if len(data) > MAX_ATTACHMENT_BYTES:
        return None, "too_large"
    return data, attachment.filename or "input.bin"


async def _mood_input(interaction, text):
    stripped = (text or "").strip()
    if stripped:
        return mood_source("", stripped)
    ref = ""
    if interaction.message is not None:
        ref = await _reference_text(interaction.channel, interaction.message.reference)
    return mood_source(ref, "")


def _user_id(interaction: discord.Interaction) -> str:
    return str(interaction.user.id)


def _touch_user_profile(store: JobStore, interaction: discord.Interaction) -> str:
    uid = _user_id(interaction)
    name = interaction.user.display_name or interaction.user.name
    avatar = None
    if interaction.user.display_avatar:
        avatar = str(interaction.user.display_avatar.url)
    store.update_user_profile(uid, display_name=name, avatar_url=avatar)
    return uid


def _active_job_for_user(store: JobStore, discord_id: str) -> dict | None:
    with store._connect() as conn:
        row = conn.execute(
            """
            SELECT j.id, j.status FROM jobs j
            JOIN users u ON u.id = j.user_id
            WHERE u.discord_id = ? AND j.status IN ('queued', 'assigned', 'running')
            ORDER BY j.id DESC LIMIT 1
            """,
            (discord_id,),
        ).fetchone()
    return None if row is None else dict(row)


def _worker_has(store: JobStore, key: str) -> bool | None:
    models = store.get_worker_models()
    if models is None:
        return None
    return bool(models.get(key))


def format_queue_ack_message(info: dict) -> str:
    pos = int(info.get("position") or 1)
    ahead = int(info.get("ahead") or 0)
    eta = format_eta_ru(int(info.get("eta_sec") or 1))
    if ahead > 0:
        return (
            f"Модуль занят, ты в очереди: позиция {pos}. "
            f"Оценка ожидания: {eta}."
        )
    return f"Обрабатываю запрос… Оценка: {eta}."


async def _defer_with_queue_ack(
    interaction: discord.Interaction,
    store: JobStore,
    job_id: int,
    worker_kind: str,
):
    await interaction.response.defer(thinking=True)
    try:
        info = await asyncio.to_thread(store.estimate_queue_wait, job_id, worker_kind)
        msg = format_queue_ack_message(info)
        await interaction.edit_original_response(content=msg)
    except Exception as exc:
        log.warning("queue ack failed job_id=%s: %s", job_id, exc)


async def _wait_job(store: JobStore, job_id: int):
    return await poll_until_done(store, job_id, timeout=DEFAULT_POLL_TIMEOUT)


def _log_user_error(context: str, exc: BaseException):
    log.exception("%s: %s", context, exc)


async def _enqueue_or_reply(
    interaction, store: JobStore, discord_id: str, kind: str, payload: dict
):
    try:
        await asyncio.to_thread(_touch_user_profile, store, interaction)
        return await asyncio.to_thread(store.enqueue, discord_id, kind, payload)
    except ActiveJobError:
        await interaction.response.send_message(ACTIVE_JOB_MSG)
        return None
    except PayloadTooLargeError as exc:
        _log_user_error("enqueue payload too large", exc)
        await interaction.response.send_message(PAYLOAD_TOO_LARGE_MSG)
        return None
    except QueueFullError as exc:
        _log_user_error("enqueue queue full", exc)
        await interaction.response.send_message(QUEUE_FULL_MSG)
        return None


async def _send_job_result(interaction, result: dict):
    kind = result.get("type")
    if kind == "message":
        await interaction.followup.send(result["content"][:DISCORD_LIMIT])
        return
    if kind == "chunks":
        chunks = result["chunks"]
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
        return
    if kind == "file":
        raw = base64.b64decode(result["data_b64"])
        await interaction.followup.send(
            content=result["content"],
            file=discord.File(BytesIO(raw), filename=result["filename"]),
        )
        return
    await interaction.followup.send("Неизвестный ответ. Попробуй ещё раз.")


def build_bot(db_path=None):
    intents = discord.Intents.default()
    intents.message_content = True

    async def _no_prefix(_bot, _message):
        return []

    bot = commands.Bot(command_prefix=_no_prefix, intents=intents, help_command=None)
    store = JobStore(db_path or DEFAULT_DB, recover_running=False)
    bot.job_store = store

    ai = app_commands.Group(name="ai", description="Kizaru: текст, настроение, картинка")

    @ai.command(name="text", description="Продолжить текст: lstm, lstm-word, transformer, gpt, deepseek, bert-gen")
    @app_commands.describe(
        model="lstm — символы; lstm-word — слова; transformer; gpt; deepseek; bert-gen (MLM)",
        start="Стартовая строка (по умолчанию «Я»)",
        temp="Температура: ниже — повторы, выше — битые слова. Пусто — дефолт модели",
        lines="Сколько строк сгенерировать (1-12, по умолчанию 6)",
        length="Бюджет: символы (слова у lstm-word), для bert-gen — длина окна",
        iters="Только bert-gen: число проходов Gibbs (по умолчанию 32)",
    )
    @app_commands.choices(
        model=[
            app_commands.Choice(name="lstm", value="lstm"),
            app_commands.Choice(name="lstm-word", value="lstm-word"),
            app_commands.Choice(name="transformer", value="transformer"),
            app_commands.Choice(name="gpt", value="gpt"),
            app_commands.Choice(name="deepseek", value="deepseek"),
            app_commands.Choice(name="bert-gen", value="bert-gen"),
        ]
    )
    async def ai_text(
        interaction: discord.Interaction,
        model: app_commands.Choice[str],
        start: str = "Я",
        temp: app_commands.Range[float, TEMP_MIN, TEMP_MAX] = None,
        lines: app_commands.Range[int, LINES_MIN, LINES_MAX] = None,
        length: app_commands.Range[int, WORD_LENGTH_MIN, CHAR_LENGTH_MAX] = None,
        iters: app_commands.Range[int, ITERS_MIN, ITERS_MAX] = None,
    ):
        key = model.value
        has = _worker_has(store, key)
        if has is not True:
            await interaction.response.send_message(
                SERVICE_OFFLINE_MSG.format(name=key)
            )
            return
        payload = build_text_payload(
            key, start, temp=temp, length=length, lines=lines, iters=iters
        )
        job_id = await _enqueue_or_reply(
            interaction,
            store,
            _user_id(interaction),
            "text",
            payload,
        )
        if job_id is None:
            return
        worker_kind = resolve_worker_kind("text", payload)
        await _defer_with_queue_ack(interaction, store, job_id, worker_kind)
        try:
            result = await _wait_job(store, job_id)
        except Exception as exc:
            _log_user_error("ai text", exc)
            await interaction.followup.send(GENERIC_ERROR_MSG)
            return
        await _send_job_result(interaction, result)

    @ai.command(name="mood", description="Настроение: оси добрый/злой и веселый/грустный")
    @app_commands.describe(text="Текст; пусто — взять reply")
    async def ai_mood(interaction: discord.Interaction, text: str = ""):
        has = _worker_has(store, "mood")
        if has is not True:
            await interaction.response.send_message(
                SERVICE_OFFLINE_MSG.format(name="mood")
            )
            return
        source = await _mood_input(interaction, text)
        if not source:
            await interaction.response.send_message(
                "Укажи text или вызови команду ответом на сообщение с текстом."
            )
            return
        payload = {"source": source}
        job_id = await _enqueue_or_reply(
            interaction,
            store,
            _user_id(interaction),
            "mood",
            payload,
        )
        if job_id is None:
            return
        await _defer_with_queue_ack(
            interaction, store, job_id, resolve_worker_kind("mood", payload)
        )
        try:
            result = await _wait_job(store, job_id)
        except Exception as exc:
            _log_user_error("ai mood", exc)
            await interaction.followup.send(GENERIC_ERROR_MSG)
            return
        await _send_job_result(interaction, result)

    @ai.command(name="pic", description="simple|line|render|details; cli=файл PNG")
    @app_commands.describe(
        mode="Четыре режима: simple|line|render|details",
        file="Картинка, GIF или видео",
        cli="Ответ файлом PNG/GIF (как CLI still), не текстом",
        scale="Файл: px на квадрант смайла, по умолчанию 2",
        cells="Файл: колонки, по умолчанию 320",
        dark="simple: символ тени (по умолчанию ⬛)",
        light="simple: символ света (по умолчанию ⬜)",
        line_thr="line|render|details: порог measured line (default 0.45)",
        details_thr="details: порог soft-паттернов (default 0.08)",
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="simple", value="2"),
            app_commands.Choice(name="line", value="1"),
            app_commands.Choice(name="render", value="3"),
            app_commands.Choice(name="details", value="4"),
        ]
    )
    async def ai_pic(
        interaction: discord.Interaction,
        mode: app_commands.Choice[str],
        file: discord.Attachment,
        cli: bool = False,
        scale: app_commands.Range[int, 1, 32] = CLI_DEFAULT_SCALE,
        cells: app_commands.Range[int, 8, CLI_MAX_COLS] = CLI_DEFAULT_CELLS,
        dark: str | None = None,
        light: str | None = None,
        line_thr: float | None = None,
        details_thr: float | None = None,
    ):
        data, filename = await _attachment_bytes(file)
        if filename == "too_large":
            await interaction.response.send_message(ATTACHMENT_TOO_LARGE_MSG)
            return
        if not data:
            await interaction.response.send_message(
                "Нужно вложение: PNG, JPG, GIF или видео."
            )
            return
        key = mode.value
        if key not in ("1", "2", "3", "4"):
            await interaction.response.send_message(
                "Режим: simple, line, render или details."
            )
            return
        as_file = bool(cli) or key == "3"
        animated = None
        if as_file:
            from core.image.probe import is_animated_media

            animated = is_animated_media(data, filename)
        try:
            payload = build_pic_payload(
                key,
                data,
                filename,
                scale=max(1, int(scale)) if as_file else None,
                cells=int(cells) if as_file else None,
                animated=animated,
                as_file=as_file,
                dark=dark if key == "2" else None,
                light=light if key == "2" else None,
                line_thr=line_thr if key in ("1", "3", "4") else None,
                details_thr=details_thr if key == "4" else None,
            )
        except ValueError as exc:
            await interaction.response.send_message(str(exc))
            return
        job_id = await _enqueue_or_reply(
            interaction,
            store,
            _user_id(interaction),
            "pic",
            payload,
        )
        if job_id is None:
            return
        await _defer_with_queue_ack(
            interaction, store, job_id, resolve_worker_kind("pic", payload)
        )
        try:
            result = await _wait_job(store, job_id)
        except Exception as exc:
            _log_user_error("ai pic", exc)
            await interaction.followup.send(GENERIC_ERROR_MSG)
            return
        await _send_job_result(interaction, result)

    @ai.command(name="me", description="Статистика твоих заявок")
    async def ai_me(interaction: discord.Interaction):
        uid = await asyncio.to_thread(_touch_user_profile, store, interaction)
        stats = await asyncio.to_thread(store.user_stats_for_discord, uid)
        name = interaction.user.display_name or interaction.user.name
        embed = discord.Embed(title=f"Статистика — {name}")
        if interaction.user.display_avatar:
            embed.set_thumbnail(url=interaction.user.display_avatar.url)
        for title, value in format_me_embed_lines(stats, name):
            embed.add_field(name=title, value=value[:1024], inline=False)
        await interaction.response.send_message(embed=embed)

    @ai.command(name="cancel", description="Отменить свою активную заявку в очереди")
    async def ai_cancel(interaction: discord.Interaction):
        uid = _user_id(interaction)
        row = await asyncio.to_thread(_active_job_for_user, store, uid)
        if row is None:
            await interaction.response.send_message("Нет активной заявки для отмены.")
            return
        job_id = int(row["id"])
        ok = await asyncio.to_thread(store.cancel, job_id, uid)
        if ok:
            await interaction.response.send_message(f"Заявка #{job_id} отменена.")
        else:
            await interaction.response.send_message("Не удалось отменить заявку.")

    @ai.command(name="help", description="Справка по /ai")
    async def ai_help(interaction: discord.Interaction):
        await interaction.response.send_message(AI_HELP[:DISCORD_LIMIT])

    bot.tree.add_command(ai)

    def _persist_guilds():
        entries = []
        for guild in bot.guilds:
            count = getattr(guild, "member_count", None)
            entries.append((guild.id, guild.name or "", count))
        if entries:
            store.sync_guilds(entries)

    def _record_bot_stats():
        guilds = list(bot.guilds)
        members_total = 0
        for g in guilds:
            c = getattr(g, "member_count", None)
            if c is not None:
                members_total += int(c)
        # intents.presences off → members_online always NULL
        latency_ms = None
        try:
            latency_ms = float(bot.latency) * 1000.0
        except Exception:
            latency_ms = None
        store.insert_bot_stats(
            guilds=len(guilds),
            members_total=members_total,
            members_online=None,
            latency_ms=latency_ms,
        )
        store.set_meta(
            "discord",
            json.dumps({"pid": os.getpid(), "guilds": len(guilds)}, ensure_ascii=False),
        )

    async def _bot_stats_loop():
        await bot.wait_until_ready()
        while not bot.is_closed():
            try:
                await asyncio.to_thread(_record_bot_stats)
            except Exception as exc:
                log.warning("bot_stats failed: %s", exc)
            await asyncio.sleep(BOT_STATS_INTERVAL_SEC)

    @bot.event
    async def on_ready():
        log.info("Bot is ready as %s", bot.user)
        try:
            await asyncio.to_thread(_persist_guilds)
        except Exception as exc:
            log.warning("guild sync to db failed: %s", exc)
        if not getattr(bot, "_bot_stats_task", None) or bot._bot_stats_task.done():
            bot._bot_stats_task = asyncio.create_task(_bot_stats_loop())
        try:
            if not bot.guilds:
                log.info("slash sync: no guilds yet (invite bot to a server)")
                return
            for guild in bot.guilds:
                bot.tree.copy_global_to(guild=guild)
                guild_cmds = await bot.tree.sync(guild=guild)
                log.info(
                    "slash sync guild=%s name=%r commands=%s",
                    guild.id,
                    guild.name,
                    len(guild_cmds),
                )
            log.info(
                "slash sync: guild-only surface, %s guild(s); global sync skipped",
                len(bot.guilds),
            )
        except Exception as exc:
            log.exception("slash sync failed: %s", exc)

    return bot


def parse_args():
    parser = argparse.ArgumentParser(description="Discord bot (enqueue only; workers compute)")
    parser.add_argument("--token", default="", help="prefer DISCORD_TOKEN env")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="SQLite queue database (default: data/bot.db)",
    )
    return parser.parse_args()


def main():
    from core.logging_setup import configure_logging

    configure_logging("discord")
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)
    args = parse_args()
    token = (args.token.strip() or os.environ.get("DISCORD_TOKEN", "")).strip()
    if not token:
        raise SystemExit("Set DISCORD_TOKEN or pass --token")
    bot = build_bot(db_path=Path(args.db))
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
