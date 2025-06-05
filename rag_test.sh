# python tests/rag_test.py --dataset kiwi --chat_model Qwen/Qwen2.5-14B-Instruct 
# python tests/ragchecker_test.py --dataset kiwi --chat_model Qwen/Qwen2.5-14B-Instruct

# python tests/rag_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct --dataset kiwi 
# python tests/ragchecker_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct --dataset kiwi 

# python tests/rag_test.py --chat_model Qwen/Qwen2.5-14B-Instruct 
# python tests/ragchecker_test.py --chat_model Qwen/Qwen2.5-14B-Instruct

# python tests/rag_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct
# python tests/ragchecker_test.py --random_shuffle --chat_model Qwen/Qwen2.5-14B-Instruct 

# python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset musique
# python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset 2wikimqa_e
# python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset hotpotqa_e

export CUDA_VISIBLE_DEVICES=0,1

# python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset musique --sorted
# python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset 2wikimqa_e --sorted
python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset hotpotqa_e --sorted
python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset narrativeqa --sorted

python tests/rag_faiss.py --chat_model Qwen/Qwen3-8B --dataset narrativeqa

python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset musique
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset 2wikimqa_e
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset hotpotqa_e
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset narrativeqa

python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset musique --sorted
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset 2wikimqa_e --sorted
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset hotpotqa_e --sorted
python tests/rag_faiss.py --chat_model Qwen/Qwen3-32B --dataset narrativeqa --sorted