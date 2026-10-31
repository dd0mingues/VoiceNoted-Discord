import logging
import os
import sys
import discord
from dotenv import load_dotenv
from groq import AsyncGroq

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TranscribeBot")

# Load environment variables
load_dotenv()

# Global variables for clients
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client: AsyncGroq | None = None


def validate_environment() -> None:
    """Validates required environment variables at startup and exits gracefully if missing."""
    global groq_client, DISCORD_TOKEN, GROQ_API_KEY
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not DISCORD_TOKEN or not GROQ_API_KEY:
        missing_vars = []
        if not DISCORD_TOKEN:
            missing_vars.append("DISCORD_TOKEN")
        if not GROQ_API_KEY:
            missing_vars.append("GROQ_API_KEY")

        logger.critical(
            "ERRO DE CONFIGURAÇÃO: As seguintes variáveis de ambiente obrigatórias não foram encontradas: %s.\n"
            "Crie ou edite o ficheiro .env no diretório raiz do projeto e defina os valores necessários.\n"
            "Consulte o ficheiro .env.example como modelo de referência.",
            ", ".join(missing_vars),
        )
        sys.exit(1)

    groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# Constants
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB Groq Whisper API limit
VOICE_MESSAGE_FLAG = 8192  # Discord's IS_VOICE_MESSAGE bitflag (1 << 13)
AUDIO_EXTENSIONS = (".ogg", ".mp3", ".wav", ".m4a")

# Context prompt for European Portuguese (PT-PT) with WoW/gaming terminology
TRANSCRIPTION_PROMPT = (
    "Transcrição em português de Portugal (PT-PT) de uma conversa entre gamers sobre World of Warcraft (WoW). "
    "Manter termos em inglês quando ditos: Guild, Guild Master, GM, Raid, Raid Leader, RL, PUG, pugar, "
    "loot, loot drama, wipe, boss, aggro, pull, tank, healer, DPS, mythic, M+, dungeon, addon, Discord. "
    "Calão português: gajos, malta, cenas, fixe, caralho, caraças, acumulado."
)

# Common Whisper hallucinations on silent or background-noise-only audio
HALLUCINATION_PHRASES = {
    "obrigado.",
    "obrigado",
    "obrigada.",
    "obrigada",
    "muito obrigado.",
    "muito obrigado",
    "legendas pela comunidade...",
    "legendas pela comunidade",
    "legendas pela comunidade amara.org",
    "subtítulos realizados pela comunidade...",
    "subtítulos realizados pela comunidade",
    "transcrição pela comunidade...",
    "transcrição pela comunidade",
    "transcrição: ",
    "transcrição:",
    "...",
    ".",
}


def split_text(text: str, max_length: int = 1900) -> list[str]:
    """
    Splits long transcripts cleanly on sentence boundaries (. , ! , ? , \\n)
    or spaces without cutting words in half, ensuring messages stay within Discord limits.
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        segment = remaining[:max_length]
        split_pos = -1

        # 1. Search for sentence boundaries (. , ! , ? , \n)
        for sep in ("\n", ". ", "! ", "? "):
            pos = segment.rfind(sep)
            if pos != -1:
                candidate_pos = pos + len(sep)
                if candidate_pos > split_pos:
                    split_pos = candidate_pos

        # 2. If no sentence boundary found, search for space
        if split_pos <= 0:
            split_pos = segment.rfind(" ")

        # 3. Fallback: hard cut if a single continuous token exceeds max_length
        if split_pos <= 0:
            split_pos = max_length

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def is_audio_attachment(attachment: discord.Attachment) -> bool:
    """Checks if an attachment has an audio content type or audio file extension."""
    if attachment.content_type and attachment.content_type.startswith("audio/"):
        return True
    if attachment.filename and attachment.filename.lower().endswith(AUDIO_EXTENSIONS):
        return True
    return False


def get_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """
    Identifies audio attachments in a message, checking both the
    Discord IS_VOICE_MESSAGE bitflag (8192) and standard audio attachments.
    """
    if not message.attachments:
        return []

    is_voice_message = bool(message.flags.value & VOICE_MESSAGE_FLAG)
    audio_attachments = []

    for attachment in message.attachments:
        if is_voice_message or is_audio_attachment(attachment):
            audio_attachments.append(attachment)

    return audio_attachments


# Configure Discord client intents
# Explicitly disable voice_states so the bot never connects to voice channels
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.voice_states = False

class VoiceNotedClient(discord.Client):
    """Discord client with optional background health-check HTTP server for cloud platforms (e.g. Render)."""

    async def setup_hook(self) -> None:
        port = os.getenv("PORT")
        if port:
            try:
                from aiohttp import web

                async def handle_health(request: web.Request) -> web.Response:
                    return web.Response(
                        text="VoiceNoted Discord Bot is active!\nStatus: 200 OK",
                        content_type="text/plain",
                    )

                app = web.Application()
                app.router.add_get("/", handle_health)
                app.router.add_get("/health", handle_health)

                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", int(port))
                await site.start()
                logger.info("Servidor HTTP de health-check ativo na porta %s (Render / Cloud)", port)
            except Exception as e:
                logger.exception("Erro ao iniciar servidor HTTP de health-check na porta %s: %s", port, e)


client = VoiceNotedClient(intents=intents)


@client.event
async def on_ready() -> None:
    """Invoked when the bot has connected to Discord."""
    logger.info("==================================================")
    logger.info("Bot autenticado como: %s (ID: %s)", client.user.name, client.user.id)
    logger.info("Conectado a %d servidor(es)", len(client.guilds))
    logger.info("Intents configurados - message_content: True, voice_states: False")
    logger.info("Pronto para transcrever notas de voz (PT-PT)!")
    logger.info("==================================================")


@client.event
async def on_message(message: discord.Message) -> None:
    """Processes incoming messages to detect and transcribe voice notes and audio."""
    # Ignore messages sent by bots (including our own messages)
    if message.author.bot:
        return

    # Check for voice messages or audio attachments
    audio_attachments = get_audio_attachments(message)
    if not audio_attachments:
        return

    for attachment in audio_attachments:
        # Check size before downloading to conserve bandwidth and respect Groq's 25MB limit
        if attachment.size and attachment.size > MAX_AUDIO_BYTES:
            logger.warning(
                "Anexo de áudio '%s' na mensagem %s excede 25MB (%d bytes)",
                attachment.filename,
                message.id,
                attachment.size,
            )
            await message.reply(
                "⚠️ O áudio excede o limite de 25MB da API.",
                mention_author=False,
            )
            continue

        try:
            # Show typing indicator in the channel while downloading & transcribing
            async with message.channel.typing():
                audio_bytes = await attachment.read()

                # Guard check against downloaded payload size
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    await message.reply(
                        "⚠️ O áudio excede o limite de 25MB da API.",
                        mention_author=False,
                    )
                    continue

                filename = attachment.filename or "voice.ogg"
                content_type = attachment.content_type or "audio/ogg"

                logger.info(
                    "A transcrever áudio da mensagem %s (autor: %s, ficheiro: %s, tamanho: %d bytes)",
                    message.id,
                    message.author,
                    filename,
                    len(audio_bytes),
                )

                # Send in-memory buffer to Groq Whisper Large-v3 (strictly transcriptions, never translations)
                transcription = await groq_client.audio.transcriptions.create(
                    file=(filename, audio_bytes, content_type),
                    model="whisper-large-v3",
                    language="pt",
                    temperature=0.0,
                    prompt=TRANSCRIPTION_PROMPT,
                )

                raw_text = transcription.text.strip() if transcription.text else ""

                # Anti-hallucination guard: silent audio or empty text
                if not raw_text:
                    logger.info("Transcrição vazia na mensagem %s; nenhuma resposta enviada.", message.id)
                    continue

                if raw_text.lower() in HALLUCINATION_PHRASES:
                    logger.info(
                        "Alucinação de silêncio detetada ('%s') na mensagem %s; nenhuma resposta enviada.",
                        raw_text,
                        message.id,
                    )
                    continue

                # Split long transcripts cleanly (under Discord 2000-char limit)
                chunks = split_text(raw_text, max_length=1900)
                if not chunks:
                    continue

                # Reply to the original voice message with the first chunk
                first_reply = f"🎙️ **Transcrição:**\n{chunks[0]}"
                await message.reply(first_reply, mention_author=False)

                # Send subsequent chunks sequentially as follow-ups
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)

                logger.info(
                    "Transcrição enviada com sucesso para a mensagem %s (%d bloco(s))",
                    message.id,
                    len(chunks),
                )

        except Exception as e:
            logger.exception("Erro ao transcrever anexo de áudio na mensagem %s: %s", message.id, e)
            try:
                await message.reply(
                    "⚠️ Ocorreu um erro ao processar a transcrição do áudio.",
                    mention_author=False,
                )
            except Exception:
                logger.exception("Falha ao enviar mensagem de erro no Discord para a mensagem %s", message.id)


def main() -> None:
    """Bot entry point."""
    validate_environment()
    logger.info("A iniciar Discord Transcribe Bot...")
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
