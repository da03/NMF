#include <petuum_ps_common/include/petuum_ps.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include <thread>

#include "NMFEngine.hpp"
#include "util/context.hpp"

/* Petuum Parameters */
DEFINE_string(hostfile, "", "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, "Total number of clients");
DEFINE_int32(num_worker_threads, 4, "Number of app threads in this client");
DEFINE_int32(client_id, 0, "Client ID");
DEFINE_int32(num_comm_channels_per_client, 4, 
        "number of comm channels per client");
 
/* NMF Parameters */
// Input and Output
DEFINE_string(data_file, "", "Input matrix.");
DEFINE_string(input_data_format, "", "Format of input matrix file"
        ", can be \"binary\" or \"text\".");
DEFINE_bool(is_partitioned, false, 
        "Whether or not the input file has been partitioned");
DEFINE_string(output_path, "", "Output path. Must be an existing directory.");
DEFINE_string(output_data_format, "", "Format of output matrix file"
        ", can be \"binary\" or \"text\".");
DEFINE_double(maximum_running_time, -1.0, "Maximum running hours. "
        "Valid if it takes value greater than 0."
        "App will try to terminate when running time exceeds "
        "maximum_running_time, but it will take longer time to synchronize "
        "tables on different clients and save results to disk.");
DEFINE_bool(load_cache, false, "Whether or not to load B and S from cache file"
        " in cache_dirname");
DEFINE_string(cache_path, "", "Valid if load_cache is set to true. "
        "Determine the path of directory containing cache to load B and S.");

// Objective function parameters
DEFINE_int32(m, 0, "Number of rows in input matrix. ");
DEFINE_int32(n, 0, "Number of columns in input matrix. ");
DEFINE_int32(dictionary_size, 0, "Size of dictionary. "
        "Default value is number of columns in input matrix.");
// Optimization parameters
DEFINE_int32(num_epochs, 100, "Number of epochs"
        ", where each epoch approximately visit the whole dataset once. "
        "Default value is 0.5.");
DEFINE_int32(minibatch_size, 1, "Minibatch size for SGD. Default value is 1."); 
DEFINE_int32(num_eval_minibatch, 10, "Evaluate obj per how many minibatches. "
        "Default value is 10."); 
DEFINE_int32(num_eval_samples, 10, "Evaluate obj by sampling how many points."
        " Default value is 10."); 
DEFINE_int32(num_iter_S_per_minibatch, 10, 
        "How many iterations for S per minibatch. Default value is 10."); 
DEFINE_double(init_step_size_B, 0.5, "SGD step size for B at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");
DEFINE_double(step_size_offset_B, 100.0, "SGD step size for B at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 100.0.");
DEFINE_double(step_size_pow_B, 0.5, "SGD step size for B at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");
DEFINE_double(init_step_size_S, 0.5, "SGD step size for S at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");
DEFINE_double(step_size_offset_S, 100.0, "SGD step size for S at iteration t is"
        " init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 100.0.");
DEFINE_double(step_size_pow_S, 0.5, "SGD step size for S at iteration t is "
        "init_step_size * (step_size_offset + t)^(-step_size_pow). "
        "Default value is 0.5.");


/* Misc */
DEFINE_int32(table_staleness, 0, "Staleness for dictionary table."
        "Default value is 0.");

/* No need to change the following */
DEFINE_string(stats_path, "", "Statistics output file.");
DEFINE_string(consistency_model, "SSPPush", "SSP or SSPPush or ...");
DEFINE_int32(row_oplog_type, petuum::RowOpLogType::kDenseRowOpLog, 
        "row oplog type");
DEFINE_bool(oplog_dense_serialized, true, "dense serialized oplog");

int main(int argc, char * argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    petuum::TableGroupConfig table_group_config;
    table_group_config.num_comm_channels_per_client
      = FLAGS_num_comm_channels_per_client;
    table_group_config.num_total_clients = FLAGS_num_clients;
    // Dictionary table and loss table
    table_group_config.num_tables = 2;
    // + 1 for main()
    table_group_config.num_local_app_threads = FLAGS_num_worker_threads + 1;;
    table_group_config.client_id = FLAGS_client_id;
    
    petuum::GetHostInfos(FLAGS_hostfile, &table_group_config.host_map);
    if (FLAGS_consistency_model == "SSP") {
        table_group_config.consistency_model = petuum::SSP;
    } else if (FLAGS_consistency_model == "SSPPush") {
        table_group_config.consistency_model = petuum::SSPPush;
    } else if (FLAGS_consistency_model == "LocalOOC") {
        table_group_config.consistency_model = petuum::LocalOOC;
    } else {
        LOG(FATAL) << "Unknown consistency model: " << FLAGS_consistency_model;
    }
    // Stats
    table_group_config.stats_path = FLAGS_stats_path;
    // Configure row types
    petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >(0);

    // Start PS
    petuum::PSTableGroup::Init(table_group_config, false);

    // Load data
    STATS_APP_LOAD_DATA_BEGIN();
    NMF::NMFEngine nmf_engine;
    LOG(INFO) << "Data loaded!";
    LOG(INFO)<<"minibatch: "<<FLAGS_minibatch_size<<", S step: "<<FLAGS_init_step_size_S<<", B step: "<<FLAGS_init_step_size_B<<"S     iter: "<<FLAGS_num_iter_S_per_minibatch;
    STATS_APP_LOAD_DATA_END();

    // Create PS table
    //
    // B_table (dictionary_size by number of rows in input matrix)
    petuum::ClientTableConfig table_config;
    table_config.table_info.row_type = 0;
    table_config.table_info.table_staleness = FLAGS_table_staleness;
    table_config.table_info.row_capacity = FLAGS_m;
    // Assume all rows put into memory
    table_config.process_cache_capacity = 
        (FLAGS_dictionary_size == 0? FLAGS_n: FLAGS_dictionary_size);
    table_config.table_info.row_oplog_type = FLAGS_row_oplog_type;
    table_config.table_info.oplog_dense_serialized = 
        FLAGS_oplog_dense_serialized;
    table_config.table_info.dense_row_oplog_capacity = 
        table_config.table_info.row_capacity;
    table_config.thread_cache_capacity = 1;
    table_config.oplog_capacity = FLAGS_m;

    CHECK(petuum::PSTableGroup::CreateTable(0, table_config)) 
        << "Failed to create dictionary table";

    // loss table. Single column. Each column is loss in one iteration
    int max_client_n = ceil(float(FLAGS_n) / FLAGS_num_clients);
    int iter_minibatch = 
        ceil(float(max_client_n / FLAGS_num_worker_threads) 
                / FLAGS_minibatch_size);
    int num_eval_per_client = 
        (FLAGS_num_epochs * iter_minibatch - 1) 
          / FLAGS_num_eval_minibatch + 1;
    table_config.table_info.row_type = 0;
    table_config.table_info.table_staleness = 50;
    table_config.table_info.row_capacity = 1;
    table_config.process_cache_capacity = 
        num_eval_per_client * FLAGS_num_clients * 2;
    table_config.table_info.row_oplog_type = FLAGS_row_oplog_type;
    table_config.table_info.oplog_dense_serialized = 
        FLAGS_oplog_dense_serialized;
    table_config.table_info.dense_row_oplog_capacity = 
        table_config.table_info.row_capacity;
    table_config.thread_cache_capacity = 1;
    table_config.oplog_capacity = 100;

    CHECK(petuum::PSTableGroup::CreateTable(1, table_config))
        << "Failed to create loss table";

    petuum::PSTableGroup::CreateTableDone();
    LOG(INFO) << "Create Table Done!";

    std::vector<std::thread> threads(FLAGS_num_worker_threads);
    for (auto & thr: threads) {
        thr = std::thread(&NMF::NMFEngine::Start, std::ref(nmf_engine));
    }
    for (auto & thr: threads) {
        thr.join();
    }
    petuum::PSTableGroup::ShutDown();
    LOG(INFO) << "NMF shut down!";
    return 0;
}
