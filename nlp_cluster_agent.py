import json
import io
from contextlib import redirect_stdout
from langchain.llms.base import LLM
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import AIMessage
from cluster_agent import main


class FakeSimpleLLM(LLM):
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        return self._interpret(prompt)

    def predict_messages(self, messages, **kwargs):
        return AIMessage(content=self._interpret(messages[-1].content))

    def _interpret(self, text: str) -> str:
        text = text.lower()
        if "clusteriz" in text or "cluster" in text:
            algorithm = "dbscan" if "dbscan" in text else "kmeans"
            is_3d = "3d" in text

            # Buscar si se mencionan k clusters (hasta 10)
            k_clusters = None
            for n in range(2, 11):
                if f"{n} cluster" in text or f"{n} clúster" in text:
                    k_clusters = n
                    break

            args = {
                "path_csv": "data/iris_data_challenge.csv",
                "output_prefix": "outputs/nlp_agent/",
                "algorithm": algorithm,
                "is_3d": is_3d,
            }
            if k_clusters:
                args["k_clusters"] = k_clusters

            return json.dumps({
                "tool_calls": [{
                    "name": "ejecutar_cluster_agente",
                    "args": args
                }]
            })

        return "No entiendo la instrucción."

    @property
    def _llm_type(self) -> str:
        return "fake-simple-llm"


def ejecutar_cluster_agente(path_csv: str, output_prefix=str, algorithm: str = "kmeans", is_3d: bool = False, k_clusters: int = None) -> str:
    print(f"[INFO] Ejecutando clustering con main(): {path_csv}, algoritmo={algorithm}, is_3d={is_3d}, k_clusters={k_clusters}")
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            main(path_csv, output_prefix, is_3d=is_3d, algorithm=algorithm, k_clusters=k_clusters)
        output = buffer.getvalue()
        return "[INFO] Ejecución del agente completada exitosamente.\n" + output
    except Exception as e:
        return f"[INFO] Falló la ejecución: {e}"
    finally:
        buffer.close()


tools = [
    Tool.from_function(
        func=ejecutar_cluster_agente,
        name="ejecutar_cluster_agente",
        description="Ejecuta el script agent_cluster.py con los parámetros dados"
    )
]

llm = FakeSimpleLLM()
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

if __name__ == "__main__":
    print("Agente listo. Escribí una instrucción como:")
    print("  'Clusterizá el dataset con KMeans en 3D' o 'Clusteriza el dataset con 4 clusters de KMeans', 'Cluseriza usando DBSCAN'   ")
    while True:
        instruccion = input(">> ")
        if instruccion.lower() in ["salir", "exit", "quit"]:
            print("Saliendo.")
            break

        respuesta = agent.invoke({"input": instruccion})
        salida = respuesta.get("output") or respuesta

        try:
            parsed = json.loads(salida)
            if "tool_calls" in parsed:
                for tool_call in parsed["tool_calls"]:
                    if tool_call["name"] == "ejecutar_cluster_agente":
                        args = tool_call["args"]
                        resultado = ejecutar_cluster_agente(**args)
                        print("Resultado de ejecución:")
                        print(resultado)
                    else:
                        print(f"Herramienta desconocida: {tool_call['name']}")
            else:
                print("Respuesta:")
                print(salida)
        except Exception as e:
            print("Respuesta:")
            print(salida)
            print(f"Error al interpretar la respuesta: {e}")
