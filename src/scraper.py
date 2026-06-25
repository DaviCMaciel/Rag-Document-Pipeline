import requests
from bs4 import BeautifulSoup
import re

class WebScraper:
    def __init__(self):
        # Simulamos um navegador para evitar bloqueios simples de sites
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def scrape(self, url):
        print(f"Lendo conteúdo da URL: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status() # Verifica se a página carregou com sucesso
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove elementos irrelevantes (scripts, estilos, rodapés)
            for element in soup(['script', 'style', 'footer', 'nav', 'header']):
                element.decompose()
            
            # Extrai o texto limpo
            text = soup.get_text(separator=' ')
            
            # Limpeza básica com Regex (similar ao que você fez no extraction)
            clean_text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                "content": clean_text,
                "metadata": {"source": url}
            }
            
        except Exception as e:
            print(f"Erro ao acessar a URL {url}: {e}")
            return {"content": "", "metadata": {"source": url}}