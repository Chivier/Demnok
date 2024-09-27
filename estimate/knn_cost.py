from aws_lambda import *

def time_cost_for_scan_one_bucket(bucket_size, memory_mb, vector_length):
    # bucket_size: bucket size in MB
    # memory_mb: memory size
    # vector_length: length of each vector  
    
    vector_size = vector_length * 4
    bucket_vec_count = int(1.0 * bucket_size / vector_size)
    load_count = int(bucket_size / memory_mb)

    # 1. load data partition from s3
    # 2. compute distance
    # 3. go to step 1

    # total_load_size = bucket_size
    # total_compute_size = vector_length * bucket_vec_count * 3

    instruction_cnt_one_turn = vector_length * bucket_vec_count * 3
    time_cost_1 = get_lambda_exec_time(instruction_cnt_one_turn)
    time_cost_2 = get_s3_read_time(bucket_size)
    cost = (time_cost_1 + time_cost_2) * load_count
    return cost
    
def s3_cost_for_scan_one_bucket(bucket_size):
    return get_cost_for_s3_read(bucket_size)

def get_cost_for_knn(n, k, vector_length, bucket_size, memory_mb):
    # n: number of data points
    # k: number of nearest neighbors
    # vector_length: length of each vector
    # bucket_size: bucket size in MB
    # memory_mb: memory size

    # fp32 size
    vector_size = vector_length * 4
    
    # Find top k \approx find top k bucket first, then find top k nearest neighbors
    factor = 1 # approximate factor
    one_bucket_time_cost = time_cost_for_scan_one_bucket(bucket_size, memory_mb, vector_length)

    one_bucket_s3_cost = s3_cost_for_scan_one_bucket(bucket_size)
    one_bucket_lambda_cost = get_cost_for_lambda_duration(one_bucket_time_cost, memory_mb)
    one_bucket_request_cost = get_cost_for_lambda_request_count(1)
    total_cost = k * (one_bucket_s3_cost + one_bucket_lambda_cost + one_bucket_request_cost)
    return total_cost

print(get_cost_for_knn(1000000, 10, 128, 1024, 10240))

