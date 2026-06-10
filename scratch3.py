import asyncio
from src.agents.state import initial_state
from src.agents.graph import agente
import sys

async def main():
    state = initial_state("Me puedes mostrar una imagen del manual que muestre un Tejido Conectivo Especializado, Cartílago Elástico?")
    try:
        res = await agente.ainvoke(state)
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
