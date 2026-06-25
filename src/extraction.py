from PIL import Image
import easyocr
import fitz
import numpy as np
import re
import os


class Document_Reader():
    def __init__(self):
        print("Inicializando o leitor de documentos...")
        self.reader = easyocr.Reader(['pt', 'en'], gpu = False)
    
    def clean_text(self, text):
        print("Limpando o texto extraído...")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_text(self, file_path):
      file_end = os.path.splitext(file_path)[1].lower()
      filename = os.path.basename(file_path)

      supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.pdf']

      if file_end not in supported_formats:
        print(f"Formato de arquivo não suportado: {file_end}")
        return {"content": "", "metadata": {"source": filename}}

      if file_end == '.pdf':
        content = self.extract_text_from_pdf(file_path)
      else:
        content = self.extract_text_from_image(file_path)
      
      return {
            "content": self.clean_text(content),
            "metadata": {"source": filename}
        }

    def extract_text_from_image(self, file_path):
        print(f"Extraindo texto da imagem: {file_path}")
        try:
            result = self.reader.readtext(file_path, detail = 0)
            return "\n".join(result)
        except Exception as e:
            print(f"Erro ao extrair texto da imagem: {e}")
            return ""
    
    def extract_text_from_pdf(self, file_path):
        print(f"Extraindo texto do PDF: {file_path}")
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    print(f"Aviso: Página {page.number + 1} do PDF '{file_path}' parece conter pouco ou nenhum texto nativo. Aplicando OCR nessa página...")
                    pic = page.get_pixmap()
                    img = Image.frombytes("RGB", [pic.width, pic.height], pic.samples)
                    ocr_result = self.reader.readtext(np.array(img), detail = 0)
                    if ocr_result:
                        text += " ".join(ocr_result) + "\n"
                    else:
                        print(f"Aviso: Nenhum texto extraído com OCR da página {page.number + 1} do PDF '{file_path}'.")
            doc.close()
            return text
        except Exception as e:
            print(f"Erro ao extrair texto do PDF: {e}")
            return ""