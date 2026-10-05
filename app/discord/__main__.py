import argparse
import asyncio
import base64
import logging
import os
from io import BytesIO
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from core.bus.queue import (
    MAX_ATTACHMENT_BYTES,
    ActiveJobError,
    JobStore,
    PayloadTooLargeError,
    QueueFullError,
    format_eta_ru,
    poll_until_done,
)
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
    "line и simple — текст в чат, 31 колонка. "
    "cli — файл: PNG для картинки, GIF для gif или видео. "
    "`scale` и `cells` только у cli (по умолчанию 2 и 320).\n"
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
    return await poll_until_done(store, job_id)


def _log_user_error(context: str, exc: BaseException):
    log.exception("%s: %s", context, exc)


async def _enqueue_or_reply(
    interaction, store: JobStore, discord_id: str, kind: str, payload: dict
):
    try:
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

    @ai.command(name="pic", description="line и simple — текст в чат, cli — PNG или GIF")
    @app_commands.describe(
        mode="line и simple — сетка в чат; cli — PNG или GIF",
        file="Картинка, GIF или видео",
        scale="Только для cli, по умолчанию 2",
        cells="Только для cli: колонки, по умолчанию 320",
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="line", value="1"),
            app_commands.Choice(name="simple", value="2"),
            app_commands.Choice(name="cli", value="3"),
        ]
    )
    async def ai_pic(
        interaction: discord.Interaction,
        mode: app_commands.Choice[str],
        file: discord.Attachment,
        scale: app_commands.Range[int, 1, 32] = CLI_DEFAULT_SCALE,
        cells: app_commands.Range[int, 8, CLI_MAX_COLS] = CLI_DEFAULT_CELLS,
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
        if key not in ("1", "2", "3"):
            await interaction.response.send_message("Режим: line, simple или cli.")
            return
        animated = None
        if key == "3":
            from core.image.probe import is_animated_media

            animated = is_animated_media(data, filename)
        payload = build_pic_payload(
            key,
            data,
            filename,
            scale=max(1, int(scale)) if key == "3" else None,
            cells=int(cells) if key == "3" else None,
            animated=animated,
        )
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

    @bot.event
    async def on_ready():
        print(f"Bot is ready as {bot.user}")
        try:
            await asyncio.to_thread(_persist_guilds)
        except Exception as exc:
            log.warning("guild sync to db failed: %s", exc)
        try:
            if not bot.guilds:
                print("slash sync: no guilds yet (invite bot to a server)")
                return
            for guild in bot.guilds:
                bot.tree.copy_global_to(guild=guild)
                guild_cmds = await bot.tree.sync(guild=guild)
                print(
                    f"slash sync guild={guild.id} name={guild.name!r} "
                    f"commands={len(guild_cmds)}"
                )
            print(
                f"slash sync: guild-only surface, {len(bot.guilds)} guild(s); "
                "global sync skipped"
            )
        except Exception as exc:
            print(f"slash sync failed: {exc}")

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    token = (args.token.strip() or os.environ.get("DISCORD_TOKEN", "")).strip()
    if not token:
        raise SystemExit("Set DISCORD_TOKEN or pass --token")
    bot = build_bot(db_path=Path(args.db))
    bot.run(token)


if __name__ == "__main__":
    main()
