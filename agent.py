import os 
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END


# Herramientas
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
    raise ValueError("Faltan las API KEYS en el archivo .env")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

class AgentState(TypedDict):
    query: str        # input
    context: str      # info de internet 
    answer: str       # respuesta

#nodo researcher
def researcher_node(state: AgentState):
    print("Inicializando nodo investigador: {state['query']}......")

    results = tavily.search(query=state['query'], max_results=3)
    context_data = "\n".join([r['content'] for r in results['results']])
    return {"context": context_data}

#nodo writter
def writer_node(state: AgentState):
    print("redactando......")
    prompt = f"""
    Eres un analista experto. Usa el siguiente contexto para responder a la pregunta del usuario.
    
    PREGUNTA: {state['query']}
    
    CONTEXTO INVESTIGADO:
    {state['context']}
    
    RESPUESTA (Formato Markdown profesional):
    """
    
    response = llm.invoke(prompt)
    
    # Guardamos la generación final
    return {"answer": response.content}

#grafo
workflow = StateGraph(AgentState)

#nodos
workflow.add_node("investigator", researcher_node)
workflow.add_node("writer", writer_node)

# Definimos el flujo (Las sinapsis)
workflow.set_entry_point("investigator") # Empezar aquí
workflow.add_edge("investigator", "writer") # De investigar -> pasar a escribir
workflow.add_edge("writer", END) # De escribir -> terminar

# Compilamos la aplicación
app = workflow.compile()

if __name__ == "__main__":
    pregunta = "Cuales son las ultimas novedades de la empresa NVIDIA hoy?"
    print(f"🚀 Iniciando Agente para: '{pregunta}'\n")
    
    # Invocamos el grafo
    result = app.invoke({"query": pregunta})
    
    print("\n" + "="*20 + " INFORME FINAL " + "="*20 + "\n")
    print(result['answer'])
