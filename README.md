# 🎙️ VoiceNoted-Discord

Bot de Discord de alta performance e pronto para produção que transcreve automaticamente notas de voz (*voice messages*) e anexos de áudio no Discord em **Português de Portugal (PT-PT)**, preservando jargão gamer e termos de **World of Warcraft (WoW)** usando o modelo `whisper-large-v3` da API da **Groq**.

---

## ✨ Características Principais

- 🇵🇹 **Foco em Português Europeu (PT-PT)**: Transcrição nativa de dialeto português de Portugal com calão e expressões comuns.
- ⚔️ **Vocabulário Gamer & WoW**: Prompt contextual ajustado para reconhecer termos como *Guild, Raid, RL, PUG, pugar, loot, wipe, aggro, tank, healer, DPS, M+, dungeon, addon, Discord*, etc.
- 🔒 **Sem Tradução**: Força `language="pt"` e usa estritamente o endpoint de transcrição da Groq (nunca traduz para inglês).
- 🔇 **Sem Acesso a Canais de Voz**: `intents.voice_states = False`. O bot nunca entra em canais de voz nem grava conversas em direto.
- ⚡ **Processamento 100% em Memória**: Descarrega e envia o áudio diretamente para a API sem escrever ficheiros temporários em disco.
- 💬 **Divisão Automática de Mensagens**: Transcrições longas que excedam o limite de 2.000 caracteres do Discord são divididas de forma limpa em quebras de frase sem cortar palavras.
- 🛡️ **Filtro Anti-Alucinação**: Ignora áudio silencioso e alucinações comuns do Whisper (como *"Obrigado"* ou *"Legendas pela comunidade"*).
- ⏳ **Indicador Visual de Digitação**: Ativa `message.channel.typing()` enquanto processa o áudio para dar feedback imediato aos utilizadores no canal.

---

## 📋 Pré-Requisitos

- **Python 3.11+** (ou Docker)
- Uma conta no [Discord Developer Portal](https://discord.com/developers/applications)
- Uma chave de API gratuita da [Groq](https://console.groq.com/)

---

## 🚀 Passo a Passo de Configuração

### 1. Obter a Chave da API Groq (Gratuita)

1. Aceda à consola da Groq: [console.groq.com](https://console.groq.com/).
2. Inicie sessão ou registe-se gratuitamente.
3. No menu lateral, clique em **API Keys**.
4. Clique em **Create API Key**, atribua um nome (ex: `discord-transcriber`) e copie a chave gerada (inicia com `gsk_...`).

---

### 2. Criar a Aplicação e o Bot no Discord

1. Aceda ao [Discord Developer Portal](https://discord.com/developers/applications).
2. Clique no botão **New Application** no canto superior direito.
3. Introduza um nome (ex: `TranscribeBot`) e clique em **Create**.
4. No menu lateral esquerdo, clique em **Bot**:
   - Em **Build-A-Bot**, clique em **Reset Token** e copie o seu **DISCORD_TOKEN**. Guarde-o num local seguro.
   - Desça até à secção **Privileged Gateway Intents**:
     - Ative **Message Content Intent** (obrigatório para detetar notas de voz e anexos de áudio).
     - *Nota:* Não precisa de ativar o Server Members Intent nem Presence Intent.
     - Clique em **Save Changes**.

---

### 3. Gerar o Link de Convite do Bot com Permissões Exatas

1. No menu lateral esquerdo, vá a **OAuth2** ➔ **URL Generator**.
2. Na secção **Scopes**, selecione:
   - `bot`
3. Na secção **Bot Permissions** que surge em baixo, selecione apenas:
   - **View Channels** (`1024`)
   - **Send Messages** (`2048`)
   - **Send Messages in Threads** (`68719476736` - opcional se usar tópicos)
   - **Read Message History** (`65536`)
4. Copie o URL gerado no fundo da página (exemplo de formato):
   ```text
   https://discord.com/oauth2/authorize?client_id=SEU_CLIENT_ID&permissions=68608&scope=bot
   ```
5. Abra o link no seu navegador, selecione o servidor Discord onde pretende adicionar o bot e confirme a autorização.

---

### 4. Configurar as Variáveis de Ambiente

Copie o ficheiro `.env.example` para `.env`:

```bash
# No Windows PowerShell:
Copy-Item .env.example .env

# No Linux / macOS:
cp .env.example .env
```

Abra o ficheiro `.env` e cole as suas credenciais:

```env
DISCORD_TOKEN=coloque_aqui_o_seu_token_do_discord
GROQ_API_KEY=coloque_aqui_a_sua_chave_groq_gsk_...
```

---

## 💻 Executar Localmente

### Windows (PowerShell)

```powershell
# 1. Criar ambiente virtual
python -m venv .venv

# 2. Ativar ambiente virtual
.\.venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Iniciar o bot
python bot.py
```

### Linux / macOS

```bash
# 1. Criar ambiente virtual
python3 -m venv .venv

# 2. Ativar ambiente virtual
source .venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Iniciar o bot
python bot.py
```

---

## 🐳 Executar via Docker (Alojamento 24/7)

Para hospedar o bot continuamente numa VPS (Oracle Cloud, DigitalOcean, Hetzner, AWS, etc.):

```bash
# 1. Construir a imagem Docker
docker build -t transcribe-bot .

# 2. Executar o contentor em background com reinício automático
docker run -d \
  --name transcribe-bot \
  --env-file .env \
  --restart unless-stopped \
  --memory 512m \
  transcribe-bot

# 3. Verificar os logs em tempo real
docker logs -f transcribe-bot
```

Para parar o bot:
```bash
docker stop transcribe-bot
```

---

## ☁️ Alojamento Gratuito 24/7 no Render.com

O bot inclui um servidor HTTP de health-check integrado compatível com o plano gratuito do Render:

1. No [Render Dashboard](https://dashboard.render.com/), clique em **New ➔ Web Service**.
2. Conecte o repositório privado **VoiceNoted-Discord** do seu GitHub.
3. Configure os seguintes campos:
   - **Language / Environment**: `Docker`
   - **Branch**: `main`
   - **Instance Type**: `Free`
4. Na secção **Environment Variables**, adicione:
   - `DISCORD_TOKEN` = *(o seu token do Discord)*
   - `GROQ_API_KEY` = *(a sua chave da Groq)*
5. Clique em **Deploy Web Service**.
6. *(Opcional para manter 100% ativo)*: Crie um monitor gratuito em [UptimeRobot](https://uptimerobot.com/) a fazer ping no URL do Render a cada 5 minutos.


---

## ⚙️ Detalhes de Funcionamento

1. **Deteção de Áudio**:
   - Verifica a flag de mensagem `IS_VOICE_MESSAGE` (bit `8192`) emitida pelas notas de voz oficiais do Discord móvel e desktop.
   - Verifica também ficheiros de áudio anexados com extensões `.ogg`, `.mp3`, `.wav` ou `.m4a` e tipos MIME `audio/*`.

2. **Limite de 25MB**:
   - A API da Groq suporta ficheiros até 25MB. Ficheiros que ultrapassem este limite recebem uma resposta educada informativa sem bloquear o bot.

3. **Divisão de Transcrições Longas**:
   - Se uma nota de voz resultar num texto com mais de 1.900 caracteres, a função `split_text()` divide a transcrição por frases (`. `, `! `, `? `, `\n`) ou espaços.
   - O primeiro bloco é enviado em resposta direta (`reply`) com o prefixo `🎙️ **Transcrição:**\n`, e os blocos seguintes são enviados sequencialmente no mesmo canal.

---

## 📄 Licença

Este projeto é disponibilizado sob a licença MIT.
