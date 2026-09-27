import streamlit as st
import sqlite3
import json
import ast
import operator
import re
from datetime import datetime

import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI RAG Lab - Advanced Agent",
    page_icon="🤖",
    layout="wide"
)

MODEL_NAME = "openai/gpt-oss-120b"
DATABASE_FILE = "agent_memory.db"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if "last_plan" not in st.session_state:
    st.session_state.last_plan = []

if "last_verification" not in st.session_state:
    st.session_state.last_verification = ""

if "conversation" not in st.session_state:
    st.session_state.conversation = []


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_FILE)


def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# MEMORY FUNCTIONS
# ============================================================

def save_memory(memory_text):

    if not memory_text.strip():
        return "Memory cannot be empty."

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO memories (memory, created_at)
        VALUES (?, ?)
        """,
        (
            memory_text.strip(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()

    return "Memory saved successfully."


def search_memory(query):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT memory, created_at
        FROM memories
        WHERE memory LIKE ?
        ORDER BY id DESC
        LIMIT 10
        """,
        (f"%{query}%",)
    )

    results = cursor.fetchall()

    conn.close()

    if not results:
        return "No matching memories were found."

    output = []

    for memory, created_at in results:
        output.append(
            f"- {memory} ({created_at})"
        )

    return "\n".join(output)


def get_all_memories():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, memory, created_at
        FROM memories
        ORDER BY id DESC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


def clear_memories():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM memories")

    conn.commit()
    conn.close()


# ============================================================
# SAFE CALCULATOR
# ============================================================

allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}


def safe_calculate(expression):

    try:

        expression = expression.strip()

        if len(expression) > 200:
            return "Error: expression is too long."

        tree = ast.parse(expression, mode="eval")

        def evaluate(node):

            if isinstance(node, ast.Constant):

                if isinstance(node.value, (int, float)):
                    return node.value

                raise ValueError("Invalid constant.")

            if isinstance(node, ast.BinOp):

                left = evaluate(node.left)
                right = evaluate(node.right)

                operation = allowed_operators.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed.")

                return operation(left, right)

            if isinstance(node, ast.UnaryOp):

                value = evaluate(node.operand)

                operation = allowed_operators.get(type(node.op))

                if operation is None:
                    raise ValueError("Operator not allowed.")

                return operation(value)

            raise ValueError("Invalid mathematical expression.")

        result = evaluate(tree.body)

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


def extract_txt(file):

    return file.read().decode("utf-8", errors="ignore")


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + CHUNK_SIZE, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vector_database(chunks):

    if not chunks:
        return None, None

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings


# ============================================================
# RAG SEARCH
# ============================================================

def rag_search(query, top_k=5):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return "No documents have been uploaded yet."

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = st.session_state.index.search(
        query_embedding,
        min(top_k, len(st.session_state.chunks))
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index == -1:
            continue

        results.append(
            {
                "score": float(score),
                "text": st.session_state.chunks[index]
            }
        )

    if not results:
        return "No relevant information was found."

    formatted = []

    for i, result in enumerate(results, 1):

        formatted.append(
            f"""
SOURCE {i}
Similarity: {result['score']:.3f}

{result['text']}
"""
        )

    return "\n".join(formatted)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    if "GROQ_API_KEY" not in st.secrets:

        return None

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [

    {
        "type": "function",

        "function": {

            "name": "rag_search",

            "description":
            "Search uploaded documents for relevant information. "
            "Use this when the user's question may be answered "
            "from uploaded PDFs, DOCX files, or TXT files.",

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description":
                        "The information to search for."
                    },

                    "top_k": {
                        "type": "integer",
                        "description":
                        "Number of relevant chunks to retrieve."
                    }

                },

                "required": ["query"]

            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description":
            "Perform mathematical calculations. "
            "Always use this tool when accurate arithmetic is required.",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {
                        "type": "string",
                        "description":
                        "Mathematical expression to calculate."
                    }

                },

                "required": ["expression"]

            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "save_memory",

            "description":
            "Save useful non-sensitive information that the user "
            "explicitly asks the assistant to remember.",

            "parameters": {

                "type": "object",

                "properties": {

                    "memory": {
                        "type": "string",
                        "description":
                        "Information that should be remembered."
                    }

                },

                "required": ["memory"]

            }
        }
    },

    {
        "type": "function",

        "function": {

            "name": "search_memory",

            "description":
            "Search information previously stored in persistent memory.",

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description":
                        "Information to search for in memory."
                    }

                },

                "required": ["query"]

            }
        }
    }

]


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(tool_name, arguments):

    if tool_name == "rag_search":

        query = arguments.get("query", "")

        top_k = arguments.get("top_k", 5)

        result = rag_search(
            query,
            top_k
        )

        return result

    elif tool_name == "calculator":

        expression = arguments.get(
            "expression",
            ""
        )

        return safe_calculate(
            expression
        )

    elif tool_name == "save_memory":

        memory = arguments.get(
            "memory",
            ""
        )

        return save_memory(
            memory
        )

    elif tool_name == "search_memory":

        query = arguments.get(
            "query",
            ""
        )

        return search_memory(
            query
        )

    return "Unknown tool."


# ============================================================
# AGENT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """

You are an advanced AI research and productivity agent.

You have access to four tools:

1. rag_search
2. calculator
3. save_memory
4. search_memory

Your job is to solve the user's request intelligently.

IMPORTANT RULES:

- Do not invent information from uploaded documents.
- Use RAG when the answer should come from uploaded documents.
- Use calculator for mathematical calculations.
- Use memory search when the user asks about information previously remembered.
- Save memory only when the user explicitly asks you to remember something,
  or when they provide useful non-sensitive information that is clearly
  intended for future use.
- Never claim that something was remembered unless the save_memory tool
  successfully saved it.
- You may use multiple tools in one task.
- After receiving a tool result, decide whether another tool is required.
- If enough information is available, provide the final answer.
- Keep answers clear and useful.
- Explain important reasoning briefly, but do not reveal private chain-of-thought.
- When using uploaded documents, distinguish document information from
  general knowledge.

You are an agent, not simply a chatbot.

"""


# ============================================================
# PLANNING
# ============================================================

def create_plan(client, user_query):

    planning_prompt = f"""

Create a short execution plan for this user request:

{user_query}

Return ONLY a numbered list.

Example:

1. Search uploaded documents.
2. Calculate the required values.
3. Compare the results.
4. Provide the final answer.

Do not perform the task.
Only create the plan.
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content":
                "You are a task planning assistant."
            },
            {
                "role": "user",
                "content": planning_prompt
            }
        ],

        temperature=0.2

    )

    plan = response.choices[0].message.content

    return plan


# ============================================================
# VERIFICATION
# ============================================================

def verify_answer(client, user_query, answer):

    verification_prompt = f"""

Verify the following answer.

USER QUESTION:
{user_query}

ANSWER:
{answer}

Check:

1. Does the answer address the question?
2. If document information was used, is it consistent with the retrieved information?
3. Are calculations reasonable?
4. Does the answer contain obvious unsupported claims?

Return:

STATUS: PASS

or

STATUS: NEEDS_REVIEW

Then give one short explanation.

Do not rewrite the complete answer.
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content":
                "You are an answer verification assistant."
            },
            {
                "role": "user",
                "content":
                verification_prompt
            }
        ],

        temperature=0.1

    )

    return response.choices[0].message.content


# ============================================================
# ADVANCED AGENT
# ============================================================

def run_agent(client, user_query, max_iterations):

    st.session_state.agent_logs = []

    st.session_state.last_plan = []

    st.session_state.last_verification = ""

    # --------------------------------------------------------
    # STEP 1: CREATE PLAN
    # --------------------------------------------------------

    st.session_state.agent_logs.append(
        "🧠 Creating task plan..."
    )

    plan = create_plan(
        client,
        user_query
    )

    st.session_state.last_plan = plan.split("\n")

    st.session_state.agent_logs.append(
        "✅ Plan created."
    )

    # --------------------------------------------------------
    # INITIAL MESSAGES
    # --------------------------------------------------------

    memories = get_all_memories()

    memory_context = ""

    if memories:

        memory_context = "\n\nStored memory context:\n"

        for _, memory, _ in memories[:10]:

            memory_context += (
                f"- {memory}\n"
            )

    messages = [

        {
            "role": "system",
            "content":
            SYSTEM_PROMPT +
            memory_context
        },

        {
            "role": "user",
            "content": user_query
        }

    ]

    # --------------------------------------------------------
    # AGENT LOOP
    # --------------------------------------------------------

    for iteration in range(max_iterations):

        st.session_state.agent_logs.append(
            f"🔄 Agent iteration {iteration + 1}/{max_iterations}"
        )

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=messages,

            tools=TOOLS,

            tool_choice="auto",

            temperature=0.2

        )

        assistant_message = response.choices[0].message

        tool_calls = assistant_message.tool_calls

        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        if not tool_calls:

            final_answer = (
                assistant_message.content
                or "The agent did not return an answer."
            )

            st.session_state.agent_logs.append(
                "💬 Agent produced final answer."
            )

            # ------------------------------------------------
            # VERIFY
            # ------------------------------------------------

            st.session_state.agent_logs.append(
                "✅ Running final verification..."
            )

            verification = verify_answer(
                client,
                user_query,
                final_answer
            )

            st.session_state.last_verification = verification

            st.session_state.agent_logs.append(
                "🔍 Verification completed."
            )

            return final_answer

        # ----------------------------------------------------
        # ADD ASSISTANT TOOL REQUEST
        # ----------------------------------------------------

        messages.append(
            assistant_message
        )

        # ----------------------------------------------------
        # EXECUTE TOOLS
        # ----------------------------------------------------

        for tool_call in tool_calls:

            tool_name = (
                tool_call.function.name
            )

            raw_arguments = (
                tool_call.function.arguments
            )

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except Exception:

                arguments = {}

            st.session_state.agent_logs.append(
                f"🔧 Tool selected: `{tool_name}`"
            )

            st.session_state.agent_logs.append(
                f"📥 Arguments: `{arguments}`"
            )

            result = execute_tool(
                tool_name,
                arguments
            )

            st.session_state.agent_logs.append(
                f"📤 Tool completed: `{tool_name}`"
            )

            # ------------------------------------------------
            # SEND TOOL RESULT BACK TO MODEL
            # ------------------------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id":
                tool_call.id,

                "name":
                tool_name,

                "content":
                str(result)

            })

    # --------------------------------------------------------
    # MAX ITERATIONS
    # --------------------------------------------------------

    st.session_state.agent_logs.append(
        "🛑 Maximum agent iterations reached."
    )

    return (
        "The agent reached its maximum number of steps "
        "before completing the task. Please try a simpler "
        "question or increase the iteration limit."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ AI RAG Lab")

    st.caption(
        "Advanced Agentic AI System"
    )

    st.divider()

    st.subheader("📊 System Status")

    memory_count = len(
        get_all_memories()
    )

    st.write(
        f"📚 Documents: "
        f"**{len(st.session_state.documents)}**"
    )

    st.write(
        f"🧩 Chunks: "
        f"**{len(st.session_state.chunks)}**"
    )

    st.write(
        f"💾 Memories: "
        f"**{memory_count}**"
    )

    st.write(
        "🤖 Agent: **Active**"
    )

    st.write(
        "🔧 Tools: **4**"
    )

    st.divider()

    st.subheader("🛠️ Agent Settings")

    max_iterations = st.slider(
        "Maximum agent steps",
        min_value=1,
        max_value=10,
        value=6
    )

    st.divider()

    st.subheader("💾 Memory")

    if st.button(
        "🗑️ Clear All Memory",
        use_container_width=True
    ):

        clear_memories()

        st.success(
            "All memories deleted."
        )

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🤖 AI RAG Lab — Advanced Agent"
)

st.write(
    "RAG + Tool Calling + Agent + Memory + "
    "Multi-Step Workflow + Verification"
)


# ============================================================
# DASHBOARD
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "RAG",
        "Active"
    )

with col2:

    st.metric(
        "Tools",
        "4"
    )

with col3:

    st.metric(
        "Agent",
        "Active"
    )

with col4:

    st.metric(
        "Memory",
        memory_count
    )

with col5:

    st.metric(
        "Model",
        "GPT-OSS"
    )


st.divider()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header(
    "📚 Knowledge Base"
)

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=[
        "pdf",
        "docx",
        "txt"
    ],
    accept_multiple_files=True
)


if uploaded_files:

    if st.button(
        "🚀 Process Documents",
        type="primary"
    ):

        all_chunks = []

        document_names = []

        progress = st.progress(0)

        for i, file in enumerate(
            uploaded_files
        ):

            try:

                if file.name.lower().endswith(
                    ".pdf"
                ):

                    text = extract_pdf(
                        file
                    )

                elif file.name.lower().endswith(
                    ".docx"
                ):

                    text = extract_docx(
                        file
                    )

                elif file.name.lower().endswith(
                    ".txt"
                ):

                    text = extract_txt(
                        file
                    )

                else:

                    continue

                chunks = create_chunks(
                    text
                )

                all_chunks.extend(
                    chunks
                )

                document_names.append(
                    file.name
                )

                progress.progress(
                    (i + 1) /
                    len(uploaded_files)
                )

            except Exception as e:

                st.error(
                    f"Error processing "
                    f"{file.name}: {e}"
                )

        st.session_state.documents = (
            document_names
        )

        st.session_state.chunks = (
            all_chunks
        )

        (
            st.session_state.index,
            st.session_state.embeddings
        ) = build_vector_database(
            all_chunks
        )

        st.success(
            f"Processed "
            f"{len(document_names)} "
            f"documents and "
            f"{len(all_chunks)} chunks."
        )


if st.session_state.documents:

    st.success(
        "Knowledge base is ready."
    )

    with st.expander(
        "📄 Uploaded Documents"
    ):

        for document in (
            st.session_state.documents
        ):

            st.write(
                f"• {document}"
            )


# ============================================================
# AGENT CHAT
# ============================================================

st.divider()

st.header(
    "🧠 Advanced Agent"
)

user_query = st.text_area(
    "Ask the agent anything...",
    placeholder=(
        "Examples:\n"
        "• What does my uploaded document say about AI?\n"
        "• Calculate 125 * 48 and explain the result.\n"
        "• Search my memory for information about my project.\n"
        "• Compare two concepts from my uploaded document."
    ),
    height=130
)


run_agent_button = st.button(
    "🚀 Run Agent",
    type="primary",
    use_container_width=True
)


if run_agent_button:

    if not user_query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        client = get_groq_client()

        if client is None:

            st.error(
                "GROQ_API_KEY was not found. "
                "Add it to Streamlit Secrets."
            )

        else:

            with st.spinner(
                "Agent is working..."
            ):

                try:

                    answer = run_agent(
                        client,
                        user_query,
                        max_iterations
                    )

                    st.session_state.conversation.append(
                        {
                            "question":
                            user_query,

                            "answer":
                            answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Agent error: {e}"
                    )

                    answer = None

            if answer:

                st.subheader(
                    "💬 Final Answer"
                )

                st.markdown(
                    answer
                )


# ============================================================
# AGENT PLAN
# ============================================================

if st.session_state.last_plan:

    st.divider()

    st.subheader(
        "🧠 Agent Plan"
    )

    for step in (
        st.session_state.last_plan
    ):

        if step.strip():

            st.write(
                step
            )


# ============================================================
# AGENT ACTIVITY
# ============================================================

if st.session_state.agent_logs:

    st.divider()

    st.subheader(
        "🔄 Agent Activity"
    )

    with st.expander(
        "View agent execution"
    ):

        for log in (
            st.session_state.agent_logs
        ):

            st.write(
                log
            )


# ============================================================
# VERIFICATION
# ============================================================

if st.session_state.last_verification:

    st.divider()

    st.subheader(
        "✅ Answer Verification"
    )

    st.info(
        st.session_state.last_verification
    )


# ============================================================
# MEMORY PANEL
# ============================================================

st.divider()

st.header(
    "💾 Persistent Memory"
)

memories = get_all_memories()

if memories:

    for memory_id, memory, created_at in memories:

        with st.expander(
            f"Memory #{memory_id} — {created_at}"
        ):

            st.write(
                memory
            )

else:

    st.info(
        "No memories stored yet."
    )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation:

    st.divider()

    st.header(
        "📝 Recent Agent Conversations"
    )

    for item in reversed(
        st.session_state.conversation[-5:]
    ):

        with st.expander(
            item["question"][:80]
        ):

            st.markdown(
                "**Question:**"
            )

            st.write(
                item["question"]
            )

            st.markdown(
                "**Answer:**"
            )

            st.write(
                item["answer"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI RAG Lab • Stage 5 • "
    "Advanced Multi-Step Agentic Workflow"
)
