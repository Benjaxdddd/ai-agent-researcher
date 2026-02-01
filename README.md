# 🕵️‍♂️ Autonomous AI Research Agent (LangGraph + Llama 3)

> Un agente autónomo de vanguardia capaz de realizar investigación profunda en internet, redactar informes y **auto-corregirse (Self-Correcting)** mediante bucles de retroalimentación iterativos. Construido con **LangGraph**, **Tavily** y **Groq (Llama-3-70b)**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Cyclic_Architecture-orange)
![Llama 3](https://img.shields.io/badge/Model-Llama_3_70B-purple)

## 🧠 The Architecture: System 2 Thinking
A diferencia de los chatbots lineales tradicionales (DAGs), este agente implementa una **Cyclic Graph Architecture** que imita el "System 2 Thinking" humano (pensamiento lento, deliberado y correctivo).

El sistema consiste en tres nodos cognitivos especializados gestionados por una **Finite State Machine (FSM)**:

1.  **🕵️ Investigator Node:** Utiliza la **Tavily API** para realizar búsquedas web optimizadas en tiempo real. Adapta sus *queries* basándose en feedback previo.
2.  **✍️ Writer Node:** Sintetiza la información recuperada en un informe coherente y profesional utilizando **Llama-3-70b**.
3.  **⚖️ Critic Node (Reflection):** Actúa como un auditor estricto. Evalúa el informe generado contra la solicitud del usuario.
    * *Approved:* El proceso termina exitosamente.
    * *Rejected:* El Crítico genera feedback y devuelve el flujo al **Investigator**, disparando un bucle recursivo de mejora.

## ✨ Key Features
* **🔄 Self-Correction Loop:** El agente detecta información faltante o vaga y re-planifica su estrategia de investigación automáticamente sin intervención humana.
* **🛡️ Hallucination Reduction:** Las respuestas están ancladas (**Grounded**) en datos verificables recuperados de la web en tiempo real, no solo en los pesos del modelo.
* **⚡ High-Performance Inference:** Aprovecha las LPUs de Groq para pasos de razonamiento recursivo en sub-segundos.
* **🛑 Safety Brakes:** Implementa lógica para prevenir bucles infinitos (límite de iteraciones) y maneja los límites de tokens de la API mediante **Query Refinement**.

## 🛠️ Tech Stack
* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful Multi-Agent Systems)
* **LLM Engine:** Llama-3.3-70b-Versatile via [Groq](https://groq.com/)
* **Tools:** [Tavily AI](https://tavily.com/) (Search Engine for LLMs)
* **Environment:** Python 3.12, Dotenv

## 🚀 Quick Start

### 1. Clonar el Repositorio
```bash
git clone [https://github.com/TU_USUARIO/ai-agent-researcher.git](https://github.com/TU_USUARIO/ai-agent-researcher.git)
cd ai-agent-researcher