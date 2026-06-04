from src.agents.state import initial_state, AgentState


def test_initial_state():
    state = initial_state(query="Que es una osteona?")
    assert state["query"] == "Que es una osteona?"
    assert state["modo"] == "texto"
    assert state["tiene_imagen"] is False
    assert state["en_temario"] is True
    assert state["resultados_busqueda"] == {"texto": [], "imagenes": []}
    assert state["messages"] == []


def test_initial_state_with_image():
    state = initial_state(query="Que tejido es?", image_base64="abc123")
    assert state["tiene_imagen"] is True


def test_initial_state_defaults():
    state = initial_state(query="test")
    assert state["query_reescrita"] == "test"
    assert state["estructura_identificada"] == ""
    assert state["error"] == ""
    assert state["entidades"] == {"tejidos": [], "estructuras": [], "tinciones": []}


def test_state_type():
    state = initial_state(query="test")
    assert "query" in state
    assert "modo" in state
    assert "respuesta" in state
