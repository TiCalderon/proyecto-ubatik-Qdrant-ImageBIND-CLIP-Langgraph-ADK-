import base64
import tempfile
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import Config

logger = logging.getLogger(__name__)


class LLMProvider:
    _gemini_chat = None
    _gemini_vision = None
    _groq = None

    @classmethod
    def get_gemini(cls, temperature: float = None, max_tokens: int = None):
        if cls._gemini_chat is None:
            cls._gemini_chat = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=temperature or Config.LLM_TEMPERATURE,
                max_output_tokens=max_tokens or Config.LLM_MAX_TOKENS,
            )
        return cls._gemini_chat

    @classmethod
    def get_gemini_vision(cls, temperature: float = None):
        if cls._gemini_vision is None:
            cls._gemini_vision = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=Config.GOOGLE_API_KEY,
                temperature=temperature or Config.LLM_TEMPERATURE,
                max_output_tokens=2048,
            )
        return cls._gemini_vision

    @classmethod
    def get_groq(cls, temperature: float = None):
        if cls._groq is None:
            cls._groq = ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=Config.GROQ_API_KEY,
                temperature=temperature or Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
        return cls._groq

    @classmethod
    async def invoke_text(cls, system_prompt: str, user_prompt: str) -> str:
        try:
            llm = cls.get_gemini()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await llm.ainvoke(messages)
            content = response.content
            if isinstance(content, list):
                content = " ".join([p.get("text", "") for p in content if isinstance(p, dict)])
            return str(content)
        except Exception as e:
            logger.warning(f"Gemini fallo: {e}. Usando Groq...")
            llm = cls.get_groq()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await llm.ainvoke(messages)
            return response.content

    @classmethod
    async def invoke_vision(cls, system_prompt: str, user_text: str, image_base64: str) -> str:
        from langchain_core.messages import HumanMessage
        try:
            llm = cls.get_gemini_vision()
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ]
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ]
            response = await llm.ainvoke(messages)
            content = response.content
            if isinstance(content, list):
                content = " ".join([p.get("text", "") for p in content if isinstance(p, dict)])
            return str(content)
        except Exception as e:
            logger.warning(f"Gemini vision fallo: {e}")
            raise

    @classmethod
    def image_to_base64(cls, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @classmethod
    def pil_to_base64(cls, pil_image) -> str:
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
