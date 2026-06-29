import os
import re
import logging
from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup
import fitz 
from PIL import Image
import numpy as np
import easyocr
from src.utils import get_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Document_Reader():
    def __init__(self):
        logger.info("Inicializando o leitor de documentos...")

        use_gpu = get_device() != "cpu"

        self.reader = easyocr.Reader(['pt', 'en'], gpu = use_gpu)

        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def clean_text(self, text: str) -> str:
        logger.info("Limpando o texto extraído...")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
      file_end = os.path.splitext(file_path)[1].lower()
      filename = os.path.basename(file_path)

      supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.pdf']

      if file_end not in supported_formats:
        logger.warning(f"Formato de arquivo não suportado: {file_end}")
        return {"content": "", "metadata": {"source": filename}}

      if file_end == '.pdf':
        content = self.extract_text_from_pdf(file_path)
      else:
        content = self.extract_text_from_image(file_path)
      
      return {
            "content": self.clean_text(content),
            "metadata": {"source": filename}
        }
    
    def extract_text_from_url(self, url: str) -> Dict[str, Any]:
        logger.info(f"Extraindo texto da URL: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup(['script', 'style', 'footer', 'nav', 'header', 'aside']):
                element.decompose()

            text = soup.get_text(separator=' ')
            return {
                "content": self.clean_text(text),
                "metadata": {"source": url}
            }
        except requests.RequestException as e:
            logger.error(f"Erro ao acessar a URL: {e}", exc_info=True)
            return {"content": "", "metadata": {"source": url}}

    def extract_text_from_image(self, file_path: str) -> str:
        logger.info(f"Extraindo texto da imagem: {file_path}")
        try:
            result = self.reader.readtext(file_path, detail = 0)
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Erro ao extrair texto da imagem: {e}")
            return ""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        logger.info(f"Extraindo texto do PDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    logger.warning(f"Aviso: Página {page.number + 1} do PDF '{file_path}' parece conter pouco ou nenhum texto nativo. Aplicando OCR nessa página...")
                    pic = page.get_pixmap()
                    img = Image.frombytes("RGB", [pic.width, pic.height], pic.samples)
                    ocr_result = self.reader.readtext(np.array(img), detail = 0)
                    if ocr_result:
                        text += " ".join(ocr_result) + "\n"
                    else:
                        logger.warning(f"Aviso: Nenhum texto extraído com OCR da página {page.number + 1} do PDF '{file_path}'.")
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF: {e}")
            return ""