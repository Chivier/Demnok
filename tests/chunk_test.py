from demnok.utils import SimpleTextChunker

data = "./data/cs_extractions/serverlessllm/serverlessllm.md"

chunker = SimpleTextChunker()

text = chunker.file_read(data)

chunks = chunker.langchain_chunk(text)

print(chunks[20: 40])