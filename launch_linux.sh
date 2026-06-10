#!/bin/bash
echo "========================================================"
echo "   Iniciando Asistente RAG Multimodal de Histologia"
echo "========================================================"
echo ""

# 1. Verificacion de Archivo de Entorno (.env)
if [ ! -f ".env" ]; then
    echo "[INFO] Archivo .env no encontrado. Creando plantilla..."
    cp .env.example .env
    echo ""
    echo "ATENCION: Es OBLIGATORIO configurar el archivo .env."
    echo "Por favor, abre el archivo .env con un editor de texto,"
    echo "coloca tu HF_TOKEN (Hugging Face) y guarda los cambios."
    echo ""
    read -p "Presiona ENTER cuando hayas configurado el .env para continuar..."
fi

# 2. Verificacion de Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 no esta instalado."
    exit 1
fi

# 3. Verificacion de 'uv' e Instalacion
if ! command -v uv &> /dev/null; then
    echo "[INFO] Instalando gestor de dependencias rapido 'uv'..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 4. Sincronizacion de Dependencias
echo "[INFO] Verificando e instalando dependencias (esto usara la cache si ya existen)..."
uv sync

# 5. Ingesta Automatica de Base de Datos Local
if [ ! -d "qdrant_memoria" ]; then
    echo "[INFO] No se encontro la base de datos local qdrant_memoria/."
    
    # Contar PDFs
    count=$(ls -1q data/pdf/*.pdf 2>/dev/null | wc -l)
    
    if [ "$count" -gt 0 ]; then
        echo "[INFO] Se encontraron PDFs. Iniciando indexacion automatica (esto puede tomar varios minutos)..."
        uv run python -m src.ingestion.pipeline
    else
        echo "[ADVERTENCIA] No hay PDFs en data/pdf/. El sistema arrancara sin base de conocimiento."
    fi
fi

# 6. Creacion de acceso directo (Icono de escritorio) si no existe
DESKTOP_FILE="$HOME/Escritorio/Ubatik_Histologia.desktop"
if [ ! -f "$DESKTOP_FILE" ]; then
    # Some linux distros use Desktop, others Escritorio
    if [ ! -d "$HOME/Escritorio" ]; then
        DESKTOP_FILE="$HOME/Desktop/Ubatik_Histologia.desktop"
    fi
    
    echo "[INFO] Creando acceso directo en el escritorio..."
    CURRENT_DIR=$(pwd)
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Name=Asistente Histologia Ubatik
Comment=Inicia el Asistente de Histologia
Exec=gnome-terminal -- bash -c "cd '$CURRENT_DIR' && ./launch_linux.sh; exec bash"
Icon=utilities-terminal
Terminal=false
Type=Application
Categories=Education;
EOF
    chmod +x "$DESKTOP_FILE"
    # Try to copy it to standard desktop entries path as well
    mkdir -p "$HOME/.local/share/applications"
    cp "$DESKTOP_FILE" "$HOME/.local/share/applications/"
fi

# 7. Iniciar Servidor
echo "[INFO] El entorno esta listo."
echo "========================================================"
echo "  INSTRUCCIONES PARA APAGAR EL SISTEMA:"
echo "  Para detener la base de datos local y el servidor,"
echo "  SIMPLEMENTE CIERRA LA VENTANA DE ESTA TERMINAL."
echo "  (Cerrar el navegador NO detiene el sistema)."
echo "========================================================"
echo ""
echo "Iniciando servidor. Presiona Ctrl+C para apagar desde la terminal..."

# Launch browser in background after 4 seconds
(sleep 4 && xdg-open "http://localhost:10010" 2>/dev/null) &

# Run uvicorn in foreground so closing the terminal kills it
uv run uvicorn server:app --host 0.0.0.0 --port 10010
