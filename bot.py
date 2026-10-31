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
logger = logging.getLogger("VoiceNoted")

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

import re

# Context prompt template for European Portuguese (PT-PT) with WoW/gaming terminology
BASE_TRANSCRIPTION_PROMPT = (
    "Transcrição em português de Portugal (PT-PT) de uma conversa entre gamers sobre World of Warcraft (WoW). "
    "Manter termos em inglês quando ditos: Guild (escrever sempre Guild, nunca guilde), Guild Master, GM, Raid, "
    "Raid Leader, RL, PUG, pugar, loot, loot drama, wipe, boss, aggro, pull, tank, healer, DPS, mythic, M+, "
    "dungeon, addon, Discord, spreadsheet, Paint. Calão português: gajos, malta, cenas, fixe, caralho, foda-se, caraças, acumulado."
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

# In-memory registry of seen member names and nicknames per guild
guild_known_members: dict[int, set[str]] = {}


def generate_name_variants(raw_name: str) -> list[str]:
    """Generates base names and common shortened variants for a gamer username or nickname."""
    if not raw_name:
        return []

    variants: set[str] = set()

    # 1. Strip trailing numbers, underscores, and punctuation (e.g. "BrunoSilva99" -> "BrunoSilva", "staav." -> "staav")
    cleaned = re.sub(r'[\._\-\d]+$', '', raw_name).strip()
    cleaned = re.sub(r'^[\._\-]+', '', cleaned).strip()
    if len(cleaned) >= 2:
        variants.add(cleaned)

    # 2. Multi-word names: extract individual words (e.g. "Bruno Silva" -> "Bruno")
    words = raw_name.split()
    if len(words) > 1:
        for w in words:
            clean_w = re.sub(r'[\._\-\d]+', '', w).strip()
            if len(clean_w) >= 3:
                variants.add(clean_w)

    # 3. CamelCase splitting (e.g. "DarkSlayer" -> "Dark", "Slayer")
    camel_parts = re.findall(r'[A-Z][a-z0-9]+', raw_name)
    if len(camel_parts) > 1:
        for p in camel_parts:
            if len(p) >= 3:
                variants.add(p)

    # 4. Portuguese root reductions (e.g. "Caramelo" -> "Caramel")
    for v in list(variants):
        if v.lower().endswith(("elo", "ela")) and len(v) > 4:
            variants.add(v[:-1])

    return [v for v in sorted(variants) if len(v) >= 2 and not v.isdigit()]


def extract_member_aliases(member: discord.Member | discord.User) -> list[str]:
    """Extracts all recognizable display names and aliases for a Discord member."""
    if member.bot:
        return []

    candidates: set[str] = set()
    nick = getattr(member, "nick", None)
    if nick:
        candidates.add(nick)
    display_name = getattr(member, "display_name", None)
    if display_name:
        candidates.add(display_name)
    global_name = getattr(member, "global_name", None)
    if global_name:
        candidates.add(global_name)
    if member.name:
        candidates.add(member.name)

    all_variants: set[str] = set()
    for c in candidates:
        all_variants.add(c)
        for v in generate_name_variants(c):
            all_variants.add(v)

    return [v for v in sorted(all_variants) if len(v) >= 2]


def track_guild_member(guild: discord.Guild | None, member: discord.Member | discord.User) -> None:
    """Caches member names dynamically when they send messages or are seen in guilds."""
    if not guild or member.bot:
        return
    if guild.id not in guild_known_members:
        guild_known_members[guild.id] = set()

    for alias in extract_member_aliases(member):
        guild_known_members[guild.id].add(alias)


def get_transcription_prompt(guild: discord.Guild | None = None) -> str:
    """
    Constructs a contextual Whisper prompt containing WoW jargon, PT-PT slang,
    and dynamically extracted server member names and shortened nicknames.
    """
    names: set[str] = set()

    # 1. Custom names from environment variables (e.g. CUSTOM_NAMES="Caramelo, Caramel, Bruno, Rotorv")
    custom_names_env = os.getenv("CUSTOM_NAMES", "")
    if custom_names_env:
        for n in custom_names_env.split(","):
            if n.strip():
                names.add(n.strip())

    # 2. Add cached/tracked members from the guild
    if guild and guild.id in guild_known_members:
        names.update(guild_known_members[guild.id])

    # 3. Add loaded guild members if available
    if guild and guild.members:
        for m in guild.members:
            for alias in extract_member_aliases(m):
                names.add(alias)

    if names:
        # Keep top 30 aliases to stay comfortably within Whisper prompt context limits
        top_names = sorted(names)[:35]
        return f"{BASE_TRANSCRIPTION_PROMPT} Nomes dos membros da guilde/servidor: {', '.join(top_names)}."

    return BASE_TRANSCRIPTION_PROMPT


def polish_transcript(text: str) -> str:
    """Post-processes Whisper transcriptions to normalize known gaming terms."""
    # Ensure 'Guild' is written without an 'e' at the end
    return re.sub(r'\bguilde\b', 'Guild', text, flags=re.IGNORECASE)


def format_paragraphs(text: str, max_sentences_per_para: int = 3, max_chars_per_para: int = 280) -> str:
    """
    Splits continuous transcription text into readable, natural paragraphs
    separated by blank lines for clean Discord presentation.
    """
    if not text:
        return ""

    # Split text into sentences retaining punctuation
    sentence_pattern = re.compile(r'([.!?]+[\s]+)')
    parts = sentence_pattern.split(text.strip())

    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i] + parts[i + 1]
        if s.strip():
            sentences.append(s.strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    if not sentences:
        return text

    paragraphs: list[str] = []
    current_para: list[str] = []
    current_len = 0

    for sentence in sentences:
        current_para.append(sentence)
        current_len += len(sentence)
        if len(current_para) >= max_sentences_per_para or current_len >= max_chars_per_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
            current_len = 0

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)





def is_audio_attachment(attachment: discord.Attachment) -> bool:
    """Checks if an attachment has an audio content type or audio file extension."""
    if attachment.content_type and attachment.content_type.startswith("audio/"):
        return True
    if attachment.filename and attachment.filename.lower().endswith(AUDIO_EXTENSIONS):
        return True
    return False


def get_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """
    Identifies audio attachments in a message, checking:
    1. Direct attachments (including Discord voice notes with IS_VOICE_MESSAGE bitflag 8192).
    2. Forwarded message snapshots (Discord Message Forwarding feature).
    3. Referenced messages (replies to older voice notes).
    """
    audio_attachments: list[discord.Attachment] = []

    # 1. Direct attachments
    is_direct_voice = bool(message.flags.value & VOICE_MESSAGE_FLAG)
    for attachment in message.attachments:
        if is_direct_voice or is_audio_attachment(attachment):
            audio_attachments.append(attachment)

    # 2. Forwarded message snapshots (Discord Message Forwarding feature)
    snapshots = getattr(message, "message_snapshots", None)
    if snapshots:
        for snapshot in snapshots:
            snapshot_flags = getattr(snapshot, "flags", None)
            is_snapshot_voice = bool(snapshot_flags.value & VOICE_MESSAGE_FLAG) if snapshot_flags else False
            for attachment in getattr(snapshot, "attachments", []):
                if is_snapshot_voice or is_audio_attachment(attachment):
                    audio_attachments.append(attachment)

    # 3. Referenced messages (if a user replies to an existing voice message)
    if not audio_attachments and message.reference:
        ref_message = message.reference.resolved
        if isinstance(ref_message, discord.Message):
            is_ref_voice = bool(ref_message.flags.value & VOICE_MESSAGE_FLAG)
            for attachment in ref_message.attachments:
                if is_ref_voice or is_audio_attachment(attachment):
                    audio_attachments.append(attachment)

    return audio_attachments


# Configure Discord client intents
# Explicitly disable voice_states so the bot never connects to voice channels
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.voice_states = False

ENABLE_MEMBERS_INTENT = os.getenv("ENABLE_MEMBERS_INTENT", "false").lower() == "true"
if ENABLE_MEMBERS_INTENT:
    intents.members = True


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
    for guild in client.guilds:
        for member in guild.members:
            track_guild_member(guild, member)
        cached_count = len(guild_known_members.get(guild.id, set()))
        logger.info("Servidor '%s': %d nome(s)/alcunha(s) registados no vocabulário", guild.name, cached_count)

    logger.info(
        "Intents configurados - message_content: True, voice_states: False, members: %s",
        ENABLE_MEMBERS_INTENT,
    )
    logger.info("Pronto para transcrever notas de voz (PT-PT)!")
    logger.info("==================================================")


@client.event
async def on_message(message: discord.Message) -> None:
    """Processes incoming messages to detect and transcribe voice notes and audio."""
    # Ignore messages sent by bots (including our own messages)
    if message.author.bot:
        return

    # Dynamically register author names/nicknames into guild vocabulary
    if message.guild:
        track_guild_member(message.guild, message.author)

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

                # Dynamically construct context prompt with gaming terms and member names
                dynamic_prompt = get_transcription_prompt(message.guild)

                # Send in-memory buffer to Groq Whisper Large-v3 (strictly transcriptions, never translations)
                transcription = await groq_client.audio.transcriptions.create(
                    file=(filename, audio_bytes, content_type),
                    model="whisper-large-v3",
                    language="pt",
                    temperature=0.0,
                    prompt=dynamic_prompt,
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

                # Polish gaming terminology and format into readable paragraphs
                polished_text = polish_transcript(raw_text)
                formatted_text = format_paragraphs(polished_text)

                # Split long transcripts cleanly (under Discord 2000-char limit)
                chunks = split_text(formatted_text, max_length=1900)
                if not chunks:
                    continue

                # Determine header based on whether audio was forwarded or recorded directly
                is_forwarded = bool(getattr(message, "message_snapshots", None))
                if is_forwarded:
                    header = f"🎙️ **Transcrição** • *Nota Encaminhada (enviada por **{message.author.display_name}**)*"
                else:
                    header = f"🎙️ **Transcrição** • *(áudio de **{message.author.display_name}**)*"

                # Reply to the original voice message with the first chunk
                first_reply = f"{header}\n\n{chunks[0]}"
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
    logger.info("A iniciar VoiceNoted Discord Bot...")
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
