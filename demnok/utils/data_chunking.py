import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SimpleTextChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def file_read(self, data_path: str) -> str:
        with open(data_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        # Cut paragraph into sentences
        sentences = re.split(r'(?<=\.)\s*(?=[A-Z]|\n)', text)
        return sentences

    def chunk(self, data) -> List[str]:
        split_paragraphs = data.split('\n\n')
        chunk_store = []
        for paragraph in split_paragraphs:
            chunk_store.extend(self.chunk_text(paragraph))
        return chunk_store
    
    def langchain_chunk(self, data) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_text(data)
        return chunks


        
        