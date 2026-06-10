import asyncio
from src.agents.state import initial_state
from src.agents.graph import AsistenteHistologia

async def main():
    asistente = AsistenteHistologia()
    res = await asistente.chat("Me puedes mostrar alguna imagen de un Corte longitudinal y transversal de miocitos estriados voluntarios y explicar qué son?")
    print("Respuesta:", res.get("respuesta"))
    print("Imagenes detectadas:", [t for t in res.get("trayectoria", []) if t.get("nodo") == "finalizar"])
    print("Imagenes recuperadas:", [ir["etiqueta"] for ir in res.get("imagenes_recuperadas", [])])

asyncio.run(main())
