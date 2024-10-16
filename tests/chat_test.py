from demnok.agents import HFChatEngine
import torch

agent = HFChatEngine(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16)

questions = [
    "What is the capital of France?",
    "What is the capital of Germany?"
]

ans = agent.generate(questions)

print(ans)