import os, tempfile, time, pandas as pd, urllib.parse, re, unicodedata, markdown, bleach, openai as oa

from supabase import create_client
from zoneinfo import ZoneInfo
from datetime import datetime
from uuid import uuid4
from threading import Lock
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from typing import Type

from fastapi import FastAPI, Response, Request, Depends, HTTPException, APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel, Field

from openai import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader,
)
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor

# ================= CONFIG =================
def current_datetime():
    return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M")

load_dotenv()
base_dir = os.path.dirname(os.path.abspath(__file__))
manuals_path = os.path.join(base_dir, "manuais")
xlsx_path = os.path.join(base_dir, "colaboradores.xlsx")

# ================= ENTRA ID =================
BASE_URL       = os.getenv("BASE_URL", "https://assistente-faq.onrender.com").rstrip("/")
TENANT_ID      = os.getenv("TENANT_ID", "").strip()
CLIENT_ID      = os.getenv("CLIENT_ID", "").strip()
CLIENT_SECRET  = os.getenv("CLIENT_SECRET", "").strip()
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret")
REDIRECT_URI   = f"{BASE_URL}/auth/callback"
IS_LOCAL       = BASE_URL.startswith("https://assistente-faq.onrender.com")

# ================= SUPABASE =================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= FASTAPI =================
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="dc_session",
    same_site=("lax" if IS_LOCAL else "none"),
    https_only=(False if IS_LOCAL else True),
    max_age=60*60*8,
)
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Servir frontend (se existir pasta static)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# OAuth
oauth = OAuth()
oauth.register(
    name="entra",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url=f"https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration",
    redirect_uri=REDIRECT_URI,
    client_kwargs={"scope": "openid profile email offline_access"},
)

# ================= DATA =================
# Planilha de colaboradores
df = pd.read_excel(
    xlsx_path,
    dtype=str,
    usecols=["Nome", "E-mail", "Ramal", "Unidade", "Departamento", "Cargo", "Admissão"],
).fillna("Não disponível")

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    # remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

for col in df.columns:
    df[f"{col}_norm"] = df[col].map(normalize_text)

# Manuais (PDF/DOC/DOCX)
loader = DirectoryLoader(
    manuals_path,
    loader_cls=lambda path: (
        PyMuPDFLoader(path) if path.endswith(".pdf")
        else UnstructuredWordDocumentLoader(path) if path.endswith(".docx")
        else UnstructuredFileLoader(path) if path.endswith(".doc")
        else None
    ),
)
manual_docs_raw = [d for d in loader.load() if d is not None]
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
manual_docs = splitter.split_documents(manual_docs_raw)

# Vetorstore para os manuais
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(manual_docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 80})

# ================= LLM / AGENTE ÚNICO =================
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=1.5,         # +criativo (1.1–1.4 é bom)
    top_p=0.95,
    presence_penalty=0.4,    # incentiva temas novos
    frequency_penalty=0.2,   # evita repetição de frases
    streaming=True           # <- para “efeito digitação”
)

# ---- Tool customizada para DataFrame ----
class PythonInput(BaseModel):
    query: str = Field(description="Código Python a ser executado")

class CustomPythonREPLTool(BaseTool):
    name: str = "python_repl_with_df"
    description: str = (
        "Use esta ferramenta para consultar informações sobre colaboradores. "
        "Colunas disponíveis: Nome, E-mail, Ramal, Unidade, Departamento, Cargo, Admissão."
    )
    args_schema: Type[BaseModel] = PythonInput

    df: Any  # campo Pydantic

    def _run(self, query: str, run_manager=None) -> str:
        try:
            print(f"\n[DEBUG] Código recebido da LLM:\n{query}\n")

            # --- 1) Sempre trabalhar em cópia limpa ---
            base_df = self.df.copy()

            # --- 2) Aliases de colunas para evitar quebras por acento/variações ---
            #    Permite a LLM usar 'Admissao' (sem acento) e outros apelidos comuns.
            alias_cols = {
                "Admissao": "Admissão",
                "email": "E-mail",
                "Email": "E-mail",
                "departamento": "Departamento",
                "cargo": "Cargo",
                "ramal": "Ramal",
                "unidade": "Unidade",
                # *_norm já existem corretamente
            }
            for alias, real in alias_cols.items():
                if real in base_df.columns and alias not in base_df.columns:
                    base_df[alias] = base_df[real]

            scope = {
                "df": base_df,
                "pd": pd,
            }
            exec_globals, exec_locals = {**scope}, {}

            # --- 3) Auto-wrap garante _result ---
            lines = [ln for ln in (query or "").splitlines() if ln.strip()]
            if not lines:
                return "Erro: consulta vazia."
            if "_result" not in query:
                if len(lines) == 1:
                    query = f"_result = {lines[0]}"
                else:
                    lines[-1] = f"_result = {lines[-1]}"
                    query = "\n".join(lines)

            # --- 4) Executa o código enviado ---
            exec(query, exec_globals, exec_locals)

            # --- 5) Recupera o resultado com fallback ---
            result = None
            if "_result" in exec_locals:
                result = exec_locals["_result"]
            else:
                for k in ("resultados", "resultado", "res", "results", "result", "output", "out"):
                    if k in exec_locals:
                        result = exec_locals[k]
                        break

            if result is None:
                return "Executado com sucesso (mas nenhum `_result` foi definido)."

            # --- 6) Normalização de saída: DataFrame/Series -> list[dict] ---
            def norm_df(df_out: pd.DataFrame) -> list[dict]:
                # Preenche vazios e formata datas
                df_norm = df_out.copy()
                # trata NaN/NaT
                df_norm = df_norm.replace({pd.NaT: "Não disponível"}).fillna("Não disponível")

                # Formata 'Admissão' (se existir) para dd/mm/aaaa
                for col in ["Admissão", "Admissao"]:
                    if col in df_norm.columns:
                        d = pd.to_datetime(df_norm[col], errors="coerce")
                        df_norm[col] = d.dt.strftime("%d/%m/%Y")
                        df_norm[col] = df_norm[col].fillna("Não disponível")
                return df_norm.to_dict(orient="records")

            if isinstance(result, pd.DataFrame):
                payload = norm_df(result)
                print(f"[DEBUG] Resultado (DF->records), {len(payload)} linhas")
                return str(payload)

            if isinstance(result, pd.Series):
                payload = norm_df(result.to_frame().T)
                print(f"[DEBUG] Resultado (Series->record)")
                return str(payload)

            # Se já vier como list[dict], padroniza campos sensíveis
            if isinstance(result, list) and (len(result) == 0 or isinstance(result[0], dict)):
                # Normaliza datas nos dicts
                fixed = []
                for row in result:
                    row = dict(row)
                    for key in ("Admissão", "Admissao"):
                        if key in row:
                            try:
                                d = pd.to_datetime(row[key], errors="coerce")
                                row[key] = d.strftime("%d/%m/%Y") if pd.notna(d) else "Não disponível"
                            except Exception:
                                row[key] = row[key] or "Não disponível"
                    # garante "Não disponível" para None/NaN
                    for k, v in row.items():
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            row[k] = "Não disponível"
                    fixed.append(row)
                print(f"[DEBUG] Resultado (records normalizado), {len(fixed)} linhas")
                return str(fixed)

            # Demais tipos: stringifica
            print(f"[DEBUG] Resultado (raw str) -> {type(result).__name__}")
            return str(result)

        except Exception as e:
            return f"Erro: {e}"

    async def _arun(self, query: str, run_manager=None) -> str:
        raise NotImplementedError("Execução assíncrona não suportada")

python_tool = CustomPythonREPLTool(df=df)

# ---- Tool para manuais (retriever) ----
retriever_tool = create_retriever_tool(
    retriever,
    name="manuals_search",
    description="Use esta ferramenta para responder perguntas sobre manuais, políticas, normas, procedimentos, benefícios e outras informações sobre a empresa e não encontradas na planilha."
)

# ---- Agente único com múltiplas ferramentas ----

def gerar_boas_vindas(llm, user=None):
    prompt = (
        "Gere uma saudação breve e profissional para dar boas-vindas ao usuário ao assistente. "
        "Use um tom natural e cordial, mas não tãoformal, variando as palavras e um pouco do texto a cada execução. "
        "O foco deve ser sempre a empresa D. Carvalho e o esclarecimento de dúvidas, manuais, procedimentos, Intranet (colaboradores, departamentos, etc.) e outros assuntos relacionados.\n"
        "Emojis: 1 por resposta quando ajudar o tom."
    )
    resposta = llm.invoke(prompt)
    return resposta.content.strip()

tools = [python_tool, retriever_tool]

multi_agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o assistente virtual da D. Carvalho. Seu papel é ajudar colaboradores respondendo dúvidas "
     "sobre pessoas da empresa (a partir da planilha de colaboradores) e também sobre normas, políticas, programas, "
     "procedimentos e benefícios (a partir dos manuais em PDF/DOCX).\n"
     "Se perguntado sobre o funcionamente técnico do agente, explique de maneira educada o seu propósito, sem informações técnicas de código e etc, "
     "e nem forneça informações desnecessárias sobre o formato dos arquivos que você tem acesso(planilha .xlsx, docx, etc), "
     "se pergutnado, apenas diga que possue uma base de dados da empresa e que consegue consultar por lá, mão cite termos como planilha, doc, pdf, etc.\n"
     "Agora é {current_datetime}. Use essa informação quando a pergunta envolver tempo ou datas.\n\n"

    "📊 **Fontes disponíveis:**\n"
     "1. **Planilha de colaboradores (python_repl_with_df)** → contém colunas: Nome, E-mail, Ramal, Unidade, Departamento, Cargo, Admissão.\n"
     "- Use esta fonte para perguntas sobre colaboradores, setores, unidades, cargos, ramais, e-mails ou datas de admissão.\n"
     "- Todas as colunas possuem versões normalizadas com o sufixo '_norm' (sem acentos, minúsculas e sem espaços extras).\n"
     "- Para buscas simples por siglas curtas como 'ti' ou 'rh', prefira igualdade exata (==) em Departamento_norm.\n"
     "- Para nomes mais longos (ex.: 'tecnologia da informação', 'recursos humanos', 'unidade são paulo'), use .str.contains(..., case=False, na=False).\n"
     "- Quando usar .str.contains(), utilize regex com bordas de palavra, por exemplo: "
     "df['Departamento_norm'].str.contains(r'\\bti\\b', case=False, na=False).\n"
     "- Quando o usuário perguntar sobre alguma sigla como 'rh', busque pelo sinônimo 'Recursos Humanos' na planilha, por exemplo, sempre que possível.\n"
     "- Quando a dúvida envolver quantidade de pessoas em um setor ou unidade, conte o número de linhas do DataFrame filtrado.\n"
     "- Se perguntado sobre **departamento, setor ou unidade**, filtre em 'Departamento_norm' ou 'Unidade_norm'.\n"
     "- Se perguntado sobre **cargo, função ou papéis como gerente, coordenador, supervisor ou diretor**, filtre sempre em 'Cargo_norm' com .str.contains(...).\n"
     "- Se perguntado sobre 'Diretor', também considere o departamento 'Diretoria' além do cargo.\n"

     "2. **Manuais em PDF/DOCX (manuals_search)** → contém informações sobre políticas, normas, procedimentos e benefícios.\n"
     "- Use esta fonte para perguntas relacionadas a esses temas.\n\n"

    "🎭 **Estilo com autonomia**\n"
    "- Use Markdown natural. Decida onde usar **negrito** para dar foco (nomes próprios, títulos, totais, prazos, avisos). Evite negritar frases inteiras.\n"
    "- Emojis: 1–2 por resposta quando ajudar o tom.\n"
    "- Espaçamento: uma linha em branco entre **seções** ou entre **pessoas**; dentro de uma seção (campos de um mesmo item), mantenha linhas consecutivas sem linhas vazias.\n"
    "- Para listas, prefira bullets (`- item`). Se precisar de subitens, use indentação de 2 espaços.\n"
    "- Se a resposta for curta, seja mais conversacional; se for longa, seja objetivo e organize em listas.\n"
    "- Nunca invente. Se faltar dado, escreva 'Não disponível'.\n\n"

     "🔄 **Sinônimos comuns de alguns departamentos:**\n"
     "- 'rh', 'RH', 'rec humano', 'rec humanos', `recursos humanos` → 'Recursos Humanos'\n"
     "- 'tec info', 'tecnologia da informacao' → 'TI'\n"
     "- 'adm', 'administracao' → 'Administrativo'\n"
     "- 'contabil' → 'Contábil'\n"
     "- 'pos venda', 'pós venda', 'pos-venda' →  'Pós-venda'\n"
     "- 'ams', 'puk', 'PUK, 'agricultura de precisao' → 'Agricultura de Precisão'\n\n"

     "⚠️ **Regras importantes:**\n"
     "- Escolha a ferramenta correta de acordo com a pergunta.\n"
     "- Nunca invente informações: responda apenas com base na planilha ou nos manuais.\n"
     "- Use sinônimos de acordo com o contexto para encontrar e gerar a resposta da melhor forma possível.\n"
     "- 'Gerente = responsável = gestor = líder' \n."
     "- 'Departamento = setor', 'Unidade = Loja'.\n"
     "- Se referirem à unidade ou loja, e utilizarem Prudente, considere 'Presidente Prudente'.\n"
     "- Em perguntas envolvendo maior e menor setor, por exemplo, considere o número de colaboradores na planilha.\n"
     "- Hierarquia de cargos: 'Gerente' > 'Coordenador' > 'Supervisor'.\n"
     "- Se a pergunta anterior for sobre um departamento ou setor, e a pergunta atual for vaga, considere que a pergunta atual está relacionada à anterior, "
     "e caso mesmo assim o contexto não se relacione, utilize o histórico completo.\n"
     "- Exemplo: Em perguntas vagas como: 'Quem é o gerente?', use o histórico {chat_history} para buscar informações relevantes e responder de forma mais precisa, " 
     "tentando identificar sobre qual departamento ou assunto a pergunta se trata.\n"
     "- Quando usar a planilha, o resultado deve ser armazenado em `_result` e a resposta final deve ser clara e natural em português, sem mostrar código.\n"
     "- Todo código Python deve terminar com `_result = ...` (mesmo em múltiplas linhas, a última deve ser `_result = ...`).\n"
     "- Sempre que perguntado sobre nome de funcionários, sempre buscar por correspondência (df[df['Nome_norm']==), "
     "caso não seja encontrado, busque por equivalência (df[df['Nome_norm'].str.contains). Por exemplo: Nome completo: Fulano Ciclano da Silva, "
     "se o usuário perguntar se existe algum \"Fulano Ciclano\", caso não seja encontrado exatamente \"Fulano Ciclano\" (==), busque nomes que contenham essas palavras "
     "\"Fulano\", \"Ciclano\", (df = df[df['Nome_norm'].str.contains('fulano') & df['Nome_norm'].str.contains('ciclano')])"
     " ou (df[df['Nome_norm'].str.contains('felipe') | df['Nome_norm'].str.contains('pereira')]).\n\n"
     
     "🎯 **Estilo de resposta:**\n"
     "- Responda de forma natural, como se estivesse conversando com um colega de trabalho.\n"
     "- Liste nomes, e-mails, telefones, cargos de forma organizada e fácil de ler.\n"
     "- Não omita informações, se perguntado, sempre responda com o máximo de detalhes possível.\n"
     "- SEMPRE use o histórico: {chat_history} para entender o contexto da pergunta e buscar a melhor resposta.\n"
     "- Se não souber a resposta e não encontrar de forma alguma, responda de forma educada que não sabe, sem tentar inventar ou criar respostas.\n"
     "- Formate nomes de colaboradores em negrito usando Markdown (**Nome**). Use listas com - e quebras de linha entrem diferentes tópicos ou assuntos.\n"
     "- Sempre traga todas as informações possíveis, não omita dados. Sobre colaboradores, traga sempre todas as informações disponíveis na planilha colaboradores.xlsx.\n"
     "- NUNCA utilizar quebras de linha desnecessárias. No máximo 1 linha de espaço por tipo de informação.\n"
     "- Em blocos de informação, como nome, cargo e etc, utilizar apenas uma quebra de linha, sem espaços.\n"

    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
]).partial(current_datetime=current_datetime())

multi_agent_base = create_openai_functions_agent(llm=llm, tools=tools, prompt=multi_agent_prompt)
base_callbacks = [StreamingStdOutCallbackHandler()]

# ================= SESSÕES =================
def get_user_id(request: Request) -> str:
    user = request.session.get("user") or {}
    return user.get("sub") or user.get("email") or "anon"

class Session:
    __slots__ = ("id", "agent", "touched")

    def __init__(self):
        self.id = uuid4().hex

        # 1) executor base (sem memory=)
        base_executor = AgentExecutor(
            agent=multi_agent_base,
            tools=tools,
            verbose=True,
            callbacks=base_callbacks,
            name=f"FAQ D. Carvalho ({self.id[:6]})",
        )
        # 2) envelopar com histórico
    
        self.agent = RunnableWithMessageHistory(
            base_executor,
            get_session_history,
            input_messages_key="input",          # bate com seu prompt
            history_messages_key="chat_history", # bate com MessagesPlaceholder
            name=f"FAQ D. Carvalho ({self.id[:6]})",
        )
        self.touched = time.time()

SESSIONS: Dict[str, Session] = {}
SESSIONS_LOCK = Lock()
MAX_SESSIONS = 500
_HISTORY_STORE: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # session_id será algo como "userId:threadId"
    if session_id not in _HISTORY_STORE:
        _HISTORY_STORE[session_id] = ChatMessageHistory()
    return _HISTORY_STORE[session_id]

def session_key(user_id: str, thread_id: Optional[str]) -> str:
    return f"{user_id}:{thread_id or 'default'}"

def get_or_create_session(request: Request, thread_id: Optional[str]) -> Session:
    user_id = get_user_id(request)
    key = session_key(user_id, thread_id)
    now = time.time()
    with SESSIONS_LOCK:
        s = SESSIONS.get(key)
        if s is None:
            s = Session()
            if len(SESSIONS) >= MAX_SESSIONS:
                oldest_key = min(SESSIONS, key=lambda k: SESSIONS[k].touched)
                SESSIONS.pop(oldest_key, None)
            SESSIONS[key] = s
        s.touched = now
        return s

# =============== HELPERS SUPABASE ===============
def create_thread(user_id: str, title: str = "Novo chat"):
    thread_id = str(uuid4())
    supabase.table("threads").insert({
        "id": thread_id,
        "user_id": user_id,
        "title": title
    }).execute()
    return thread_id

# Salva mensagem na tabela messages
def save_message_db(thread_id: str, sender: str, text: str):
    supabase.table("messages").insert({
        "thread_id": thread_id,
        "sender": sender,
        "text": text
    }).execute()

# Lista threads do usuário
def list_threads(user_id: str):
    res = (supabase
        .table("threads")
        .select("id,title,created_at,messages!inner(id)")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute())
    # O !inner já garante existência de mensagem; se quiser, pode mapear/limpar o payload:
    rows = res.data or []
    return [{"id": r["id"], "title": r.get("title"), "created_at": r.get("created_at")} for r in rows]

# Lista mensagens de uma thread
def list_messages(thread_id: str):
    res = supabase.table("messages") \
        .select("*") \
        .eq("thread_id", thread_id) \
        .order("created_at", desc=False) \
        .execute()
    return res.data or []

def update_thread_title(thread_id: str, first_question: str):
    raw = (first_question or "Novo chat").strip()
    if len(raw) <= 50:
        title = raw
    else:
        limit = 50 - 3  # espaço para "..."
        cut = raw.rfind(" ", 0, limit)
        cut = cut if cut != -1 else limit
        title = raw[:cut].rstrip() + "..."
    supabase.table("threads").update({"title": title}).eq("id", thread_id).execute()


def delete_thread(user_id: str, thread_id: str):
    # garante que só o dono pode apagar
    supabase.table("messages").delete().eq("thread_id", thread_id).execute()
    supabase.table("threads").delete().eq("id", thread_id).eq("user_id", user_id).execute()

# ================= AUTENTICAÇÃO =================
PUBLIC_PATHS    = {"/login", "/logout", "/auth/callback", "/favicon.ico", "/whoami", "/debug/session"}
PUBLIC_PREFIXES = ("/static/",)

def is_api_request(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return "application/json" in accept or request.url.path.startswith(("/perguntar", "/new_session"))

async def require_user(request: Request):
    path = request.url.path
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return
    sess = request.session
    user = (sess or {}).get("user")
    if user:
        return
    if is_api_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    raise HTTPException(status_code=302, headers={"Location": "/login"})

protected = APIRouter(dependencies=[Depends(require_user)])

# ---------- API do chat ----------
class Pergunta(BaseModel):
    texto: str
    thread_id: str | None = None

NAME_LINE_RE = re.compile(r'^\s*-?\s*\*\*.+\*\*\s*$')  # "- **Nome**" ou "**Nome**"

def _compact_people_text(raw: str) -> str:
    raw = (raw or "").replace("\r\n", "\n")

    # 1) remove linhas 100% vazias
    lines = [ln.rstrip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln.strip() != ""]

    # 2) insere UMA linha em branco só antes de linhas de nome (menos na 1ª)
    out, first_name_seen = [], False
    for ln in lines:
        if NAME_LINE_RE.match(ln):
            if first_name_seen:
                out.append("")  # linha em branco separando pessoas
            first_name_seen = True
            out.append(ln.strip())
        else:
            out.append(ln.strip())

    return "\n".join(out)

def markdown_html(texto: str) -> str:
    t = (texto or "").replace("\r\n", "\n")
    # 1) remova espaços à direita
    t = "\n".join(l.rstrip() for l in t.split("\n"))
    # 2) limite linhas em branco consecutivas a no máximo 2
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 3) renderize markdown SEM nl2br (deixe listas funcionarem naturalmente)
    html = markdown.markdown(t, extensions=["extra", "sane_lists"])

    # 4) sanitize
    html = bleach.clean(
        html,
        tags=["p","br","strong","em","ul","ol","li","a","code","pre","h1","h2","h3","h4","h5","h6","blockquote"],
        attributes={"a": ["href","title","rel","target"]},
        strip=True,
    )

    # 5) ajuste leve: às vezes o renderer coloca <p> dentro de <li> (cria margens grandes)
    html = re.sub(r"<li>\s*<p>", "<li>", html)
    html = re.sub(r"</p>\s*</li>", "</li>", html)

    return html

def ensure_thread_exists(user_id: str, thread_id: str, title: str | None = None) -> str:
    if not thread_id:
        thread_id = str(uuid4())

    # UPSERT evita corrida + dispensa SELECT prévio
    supabase.table("threads").upsert(
        {"id": thread_id, "user_id": user_id, "title": (title or "Novo chat")},
        on_conflict="id"
        #[:60]
    ).execute()
    return thread_id

@protected.post("/perguntar")
def perguntar(p: Pergunta, request: Request) -> dict:
    user_id   = get_user_id(request)
    thread_id = p.thread_id or str(uuid4())          # fallback
    pergunta  = p.texto
    sess      = get_or_create_session(request, thread_id)

    try:
        metadata = {"User": request.session.get("user", {}).get("name"), "Email": request.session.get("user", {}).get("email") }

        # 1) Garante que a thread exista (se tiver sido apagada, recria)
        thread_id = ensure_thread_exists(user_id, thread_id, "Novo chat")

        # 2) Salva a PERGUNTA do usuário
        save_message_db(thread_id, "user", pergunta)

        # 3) Atualiza o título com a primeira pergunta (ou sempre, se preferir)
        update_thread_title(thread_id, pergunta)

        # 4) Invoca o agente
        result = sess.agent.invoke(
            {"input": pergunta},
            config={
                "metadata": metadata,
                "configurable": {"session_id": f"{user_id}:{thread_id}"},
                "run_name": f"FAQ D. Carvalho ({thread_id[:6]})",
                "current_datetime": current_datetime(),
            },
        )
        resposta = result.get("output") or result.get("output_text") or "Sem resultado."

        # 5) Salva a RESPOSTA do bot
        save_message_db(thread_id, "bot", resposta)

    except Exception as e:
        resposta = f"Ocorreu um erro: {e}"

    return {"thread_id": thread_id, "resposta": markdown_html(resposta)}

# ================= Rotas protegidas =================
@protected.get("/new_session", response_class=JSONResponse)
def new_session(request: Request):
    user = request.session.get("user")
    if not user:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    user_id = get_user_id(request)

    # cria a thread vazia (sem mensagens)
    thread_id = str(uuid4())
    ensure_thread_exists(user_id, thread_id, "Novo chat")

    # gera o welcome APENAS para exibição (não salvar)
    welcome = gerar_boas_vindas(llm)

    return JSONResponse({
        "thread_id": thread_id,
        "mensagem_inicial": welcome
    })

@protected.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Recebe audio/webm|ogg|mp3, transcreve com Whisper e retorna texto.
    Não salva o arquivo por padrão (fica só em memória/temporário).
    """
    # salva temporário (Whisper API lê stream mas usamos arquivo por simplicidade)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or ".webm")[1] or ".webm") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            tr = oa.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="pt"  # força PT-BR; pode omitir para auto
            )
        text = (tr.text or "").strip()
        return {"text": text}
    finally:
        try: os.remove(tmp_path)
        except: pass

@protected.get("/threads")
def get_threads(request: Request):
    user_id = get_user_id(request)
    threads = list_threads(user_id)
    return {"threads": threads}

@protected.get("/messages/{thread_id}")
def get_messages(thread_id: str, request: Request):
    msgs = list_messages(thread_id)
    return {"messages": msgs}

@protected.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>App</h1><p>static/index.html não encontrado.</p>", status_code=200)

@protected.get("/{path:path}", response_class=HTMLResponse)
async def spa_any(path: str, request: Request):
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>App</h1><p>static/index.html não encontrado.</p>", status_code=200)
    
@protected.delete("/threads/{thread_id}")
def remove_thread(thread_id: str, request: Request):
    user_id = get_user_id(request)
    delete_thread(user_id, thread_id)

    key = f"{user_id}:{thread_id}"
    _HISTORY_STORE.pop(key, None)
    SESSIONS.pop(key, None)

    return {"status": "ok"}

# ========= Rotas públicas =========
# servir /favicon.ico a partir da raiz do projeto
@app.get("/favicon.ico")
def favicon():
    path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    return FileResponse(path, media_type="image/x-icon")

@app.get("/login")
async def login(request: Request):
    request.session["oauth_init"] = True
    return await oauth.entra.authorize_redirect(request, redirect_uri=REDIRECT_URI)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.entra.authorize_access_token(request)
    except Exception as e:
        return PlainTextResponse(f"Erro em authorize_access_token: {type(e).__name__}: {e}", status_code=400)
    user = None
    if token.get("id_token"):
        try:
            user = await oauth.entra.parse_id_token(request, token)
        except Exception:
            user = None
    if user is None:
        try:
            user = token.get("userinfo") or await oauth.entra.userinfo(token=token)
        except Exception as e:
            return PlainTextResponse(f"Falhou id_token e /userinfo: {e}", status_code=400)
    sub   = user.get("sub")
    email = user.get("preferred_username") or user.get("email")
    name  = user.get("name") or email or sub
    if not sub:
        return PlainTextResponse(f"Sem 'sub' no usuário. Payload: {user}", status_code=400)
    request.session["user"] = {"sub": sub, "name": name, "email": email}
    try:
        request.session.modified = True
    except Exception:
        pass
    return RedirectResponse(url="/", status_code=303)

@app.get("/whoami", response_class=PlainTextResponse)
async def whoami(request: Request):
    u = request.session.get("user")
    cookies = request.headers.get("cookie", "")
    return PlainTextResponse("user=" + repr(u) + "\n" + "cookie=" + cookies)

@app.get("/debug/session", response_class=PlainTextResponse)
def debug_session(request: Request):
    return PlainTextResponse(repr(request.session))

async def _entra_end_session_url(post_logout_redirect_uri: str) -> str:
    try:
        metadata = await oauth.entra.load_server_metadata()
        end_sess = metadata.get("end_session_endpoint")
    except Exception:
        end_sess = None

    if not end_sess:
        tenant = TENANT_ID or "common"
        end_sess = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/logout"

    return f"{end_sess}?post_logout_redirect_uri={urllib.parse.quote(post_logout_redirect_uri, safe='')}"

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    post_logout = f"{BASE_URL}/login"
    idp_logout = await _entra_end_session_url(post_logout)  # <<-- await aqui
    resp = RedirectResponse(url=idp_logout, status_code=302)
    resp.delete_cookie("dc_session", path="/", httponly=True, samesite=("lax" if IS_LOCAL else "none"), secure=(False if IS_LOCAL else True))
    resp.delete_cookie("sid", path="/", httponly=False, samesite=("Lax" if IS_LOCAL else "None"), secure=(False if IS_LOCAL else True))
    return resp

# Router protegido
app.include_router(protected)

# --- run ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
