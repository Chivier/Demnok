import math

def get_cost_for_lambda_memory(memory_mb):
    # List of available memory sizes and their corresponding prices
    pricing = {
        128: 0.0000000017,
        512: 0.0000000067,
        1024: 0.0000000133,
        1536: 0.0000000200,
        2048: 0.0000000267,
        3072: 0.0000000400,
        4096: 0.0000000533,
        5120: 0.0000000667,
        6144: 0.0000000800,
        7168: 0.0000000933,
        8192: 0.0000001067,
        9216: 0.0000001200,
        10240: 0.0000001333
    }
    
    # Sort memory sizes to find the smallest that can hold the input
    sorted_memories = sorted(pricing.keys())
    
    for mem in sorted_memories:
        if memory_mb <= mem:
            return mem, pricing[mem]
    
    return None, None


def get_cost_for_s3_write(size_mb):
    # S3 Standard pricing for PUT, COPY, POST, or LIST requests
    request_cost = 0.0000005  # $0.0000005 per request
    data_cost = 0.000005 * size_mb / 1024  # $0.005 per GB
    return request_cost + data_cost

def get_cost_for_s3_read(size_mb):
    # S3 Standard pricing for GET and SELECT requests
    request_cost = 0.0000004  # $0.0000004 per request
    data_cost = 0.0000004 * size_mb / 1024  # $0.0004 per GB
    return request_cost + data_cost
    
# x86 Price
# First 6 Billion GB-seconds / month	$0.0000166667 for every GB-second	$0.20 per 1M requests
# Next 9 Billion GB-seconds / month	$0.000015 for every GB-second	$0.20 per 1M requests
# Over 15 Billion GB-seconds / month	$0.0000133334 for every GB-second	$0.20 per 1M requests

def get_cost_for_lambda_duration(time, memory_mb):
    gb_seconds = time * memory_mb / 1024
    if gb_seconds <= 6 * 10 ** 9:
        return gb_seconds * 0.0000166667
    elif gb_seconds <= 15 * 10 ** 9:
        return 6 * 10 ** 9 * 0.0000166667 + (gb_seconds - 6 * 10 ** 9) * 0.000015
    else:
        return 6 * 10 ** 9 * 0.0000166667 + 9 * 10 ** 9 * 0.000015 + (gb_seconds - 15 * 10 ** 9) * 0.0000133334

def get_cost_for_lambda_request_count(count):
    return count * 0.20 / 1000000

def get_lambda_exec_time(instruction_cnt):
    # model name	: Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz
    # on lambda I have 6 v-CPU core at most
    # avx has 8 fp32 compute units per core
    flops = 6 * 2.8 * 10 ** 9 * 8
    return instruction_cnt / flops

# Standard S3:
# Typically ranges from 30-50 MB/s for small objects
# Can reach up to 100-200 MB/s for larger objects
# S3 Transfer Acceleration:
# Can improve transfer speeds by 50-500% for long-distance transfers
# Actual speed increase depends on factors like network conditions and object size
# S3 Express One Zone:
# Offers single-digit millisecond latency
# Can achieve throughput of up to 80 Gbps (approximately 10 GB/s)
def get_s3_read_time(size_mb, type="standard"):
    if type == "standard":
        return size_mb / (40)
    elif type == "transfer_acceleration":
        return size_mb / (200)
    elif type == "express_one_zone":
        return size_mb / (80 * 1024)
