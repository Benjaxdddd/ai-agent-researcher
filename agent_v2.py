import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Herramientas
from langchain_groq import ChatGroq
from tavily import TavilyClient

# Configuración
load_dotenv()





tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)


# 1. ESTADO Recursivo
# Añadimos 'revision_number' para evitar bucles infinitos 
# Añadimos 'feedback' para que el investigador sepa qué buscar la próxima vez.

class AgentState(TypedDict):
    query: str
    context: str
    answer: str
    feedback: str
    revision_number: int

# 2. Nodo Investigador (Ahora escucha feedback)

def researcher_node(state: AgentState):
    query = state['query']
    feedback = state.get('feedback', '')
    
    # LÓGICA MEJORADA:
    if feedback:
        print(f"Re-Investigando (Intento {state['revision_number']})...")
        
        # Usamos el LLM para convertir el feedback largo en una búsqueda corta (keyword search) esto debido al limite de tabily
        prompt_refine = f"""
        Tu tarea es generar una búsqueda web CORTA (máximo 15 palabras) para Tavily.
        
        Pregunta Original: {query}
        Lo que falta (Crítica): {feedback}
        
        Genera SOLAMENTE la frase de búsqueda optimizada para encontrar lo que falta.
        """
        # Invocamos al LLM para "limpiar" la query
        response = llm.invoke(prompt_refine)
        search_query = response.content.strip()
        
        print(f"Query Refinada por IA: '{search_query}'")
    else:
        print(f"Investigando: {query}...")
        search_query = query
        
    # Safety Brake: Si el LLM alucina y genera algo muy largo, lo cortamos para que no rompa Tavily
    if len(search_query) > 300:
        search_query = search_query[:300]
        
    try:
        results = tavily.search(query=search_query, max_results=3)
        new_context = "\n".join([r['content'] for r in results['results']])
    except Exception as e:
        print(f"Error en Tavily: {e}")
        new_context = "" # Si falla, no rompemos el agente, solo seguimos sin data nueva
    
    # Cruce de contextos 
    current_context = state.get('context', '')
    return {"context": current_context + "\n\n" + new_context}

# 3. Nodo Redactor (Igual que antes)
def writer_node(state: AgentState):
    print("Redactando informe...")
    prompt = f"""
    Eres un analista experto.
    Pregunta: {state['query']}
    Contexto: {state['context']}
    
    Escribe una respuesta clara y completa.
    """
    response = llm.invoke(prompt)
    
    # Incrementamos el contador de revisiones aquí
    revision = state.get('revision_number', 0) + 1
    return {"answer": response.content, "revision_number": revision}

# 4. Nodo Crítico
def critic_node(state: AgentState):
    print("Auditando calidad del informe...")
    
    prompt = f"""
    Actúa como un editor jefe estricto. Revisa el siguiente informe para la pregunta: "{state['query']}".
    
    INFORME ACTUAL:
    {state['answer']}
    
    Si el informe es completo y responde bien, responde SOLAMENTE con la palabra: APROBADO.
    Si falta información, es vago o corto, responde con: RECHAZADO: [Explica qué falta buscar].
    """
    response = llm.invoke(prompt)
    review = response.content
    
    if "APROBADO" in review:
        return {"feedback": None} # Señal de éxito
    else:
        # Extraemos la crítica para pasarla al investigador
        feedback_clean = review.replace("RECHAZADO:", "").strip()
        return {"feedback": feedback_clean}

# 5. La Decisión de camino 
# Esta función no es un nodo, es un "semáforo" que decide el camino.

def router(state: AgentState) -> Literal["investigator", "__end__"]:
    feedback = state.get('feedback')
    revision = state.get('revision_number', 0)
    
    # Condición de Parada 1: Calidad Aprobada
    if feedback is None:
        print("Calidad Aprobada. Felicitaciones master.")
        return "__end__"
    
    # Condición de Parada 2: Límite de intentos 
    # Vital en producción para no gastar dinero infinito en bucles.
    if revision >= 3:
        print("Límite de revisiones alcanzado. Entregando lo que hay.")
        return "__end__"
    
    # Si no, volvemos a investigar
    print(f"Informe Rechazado. Volviendo a investigar...")
    return "investigator"

# --- GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("investigator", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("investigator")

workflow.add_edge("investigator", "writer")
workflow.add_edge("writer", "critic")

# Aquí está la magia cíclica: Del Crítico salimos al Router
workflow.add_conditional_edges(
    "critic",
    router,
    {
        "investigator": "investigator", # Si router dice 'investigator', ve al nodo 'investigator'
        "__end__": END                  # Si router dice '__end__', termina
    }
)

app = workflow.compile()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Una pregunta difícil que suele requerir 2 vueltas
    pregunta = "Cual es el precio de las acciones de Apple y Microsoft hoy y cual ha subido mas en las ultimas 24 horas?"
    
    print(f"Iniciando Agente Cíclico...\n")
    # Inicializamos revision_number en 0
    result = app.invoke({"query": pregunta, "revision_number": 0})
    
    print("\n" + "="*20 + " INFORME FINAL AUDITADO " + "="*20 + "\n")
    print(result['answer'])