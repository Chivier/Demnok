# python tests/rag_test.py --dataset kiwi --chat_model Qwen/Qwen2.5-14B-Instruct 
# python tests/ragchecker_test.py --dataset kiwi --chat_model Qwen/Qwen2.5-14B-Instruct

# python tests/rag_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct --dataset kiwi 
# python tests/ragchecker_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct --dataset kiwi 

# python tests/rag_test.py --chat_model Qwen/Qwen2.5-14B-Instruct 
python tests/ragchecker_test.py --chat_model Qwen/Qwen2.5-14B-Instruct

python tests/rag_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct
python tests/ragchecker_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct 