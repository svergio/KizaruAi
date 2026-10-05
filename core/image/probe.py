from io import BytesIO


def is_animated_media(data: bytes, filename: str = "") -> bool:
    name = (filename or "").lower()
    if name.endswith((".mp4", ".webm", ".mov")):
        return True
    if not data:
        return False
    if name.endswith(".gif") or data[:6] in (b"GIF87a", b"GIF89a"):
        from PIL import Image

        image = Image.open(BytesIO(data))
        try:
            image.seek(1)
            return True
        except EOFError:
            return False
    return False
