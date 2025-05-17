PYTHONWARNINGS="ignore" torchrun --nproc_per_node=4 ci_pipeline/vec_dist.py --file musique;
PYTHONWARNINGS="ignore" torchrun --nproc_per_node=4 ci_pipeline/vec_dist.py --file 2wikimqa_e;
PYTHONWARNINGS="ignore" torchrun --nproc_per_node=4 ci_pipeline/vec_dist.py --file hotpotqa_e;
PYTHONWARNINGS="ignore" torchrun --nproc_per_node=4 ci_pipeline/vec_dist.py --file narrativeqa;