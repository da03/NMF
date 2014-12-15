#include "NMFEngine.hpp"

#include <string>
#include <glog/logging.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <mutex>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "util/Eigen/Dense"
#include "util/context.hpp"

namespace NMF {

    // Constructor
    NMFEngine::NMFEngine(): thread_counter_(0) {
        // timer
        initT_ = boost::posix_time::microsec_clock::local_time();

        /* context */
        lda::Context & context = lda::Context::get_instance();
        // input and output
        data_file_ = context.get_string("data_file");
        input_data_format_ = context.get_string("input_data_format");
        is_partitioned_ = context.get_bool("is_partitioned");
        output_path_ = context.get_string("output_path");
        output_data_format_ = context.get_string("output_data_format");
        maximum_running_time_ = context.get_double("maximum_running_time");
        load_cache_ = context.get_bool("load_cache");
        cache_path_ = context.get_string("cache_path");

        // objective function parameters
        int m = context.get_int32("m");
        int n = context.get_int32("n");
        rank_ = context.get_int32("rank");

        // petuum parameters
        client_id_ = context.get_int32("client_id");
        num_clients_ = context.get_int32("num_clients");
        num_worker_threads_ = context.get_int32("num_worker_threads");

        // optimization parameters
        num_epochs_ = context.get_int32("num_epochs");
        minibatch_size_ = context.get_int32("minibatch_size");
        num_eval_minibatch_ = context.get_int32("num_eval_minibatch");
        num_eval_samples_ = context.get_int32("num_eval_samples");
        init_step_size_B_ = context.get_double("init_step_size_B");
        step_size_offset_B_ = context.get_double("step_size_offset_B");
        step_size_pow_B_ = context.get_double("step_size_pow_B");
        num_iter_S_per_minibatch_ = 
            context.get_int32("num_iter_S_per_minibatch");
        init_step_size_S_ = context.get_double("init_step_size_S");
        step_size_offset_S_ = context.get_double("step_size_offset_S");
        step_size_pow_S_ = context.get_double("step_size_pow_S");
        init_S_low_ = context.get_double("init_S_low");
        init_S_high_ = context.get_double("init_S_high");
        init_B_low_ = context.get_double("init_B_low");
        init_B_high_ = context.get_double("init_B_high");

        /* Init matrices */
        // Partition by column id mod num_clients_
        int client_n = (n - (n / num_clients_) * num_clients_ > client_id_)?
            n / num_clients_ + 1: n / num_clients_;
        // Init matrix loader of data matrix X
        if (is_partitioned_) {
            X_matrix_loader_.Init(data_file_, input_data_format_, m, client_n);
        } else {
            X_matrix_loader_.Init(data_file_, input_data_format_, m, n, 
                    client_id_, num_clients_);
        }

        // Init matrix loader of coefficients S
        if (rank_ == 0)
            rank_ = n; 
        S_matrix_loader_.Init(rank_, client_n, init_S_low_, init_S_high_);

        int max_client_n = ceil(float(n) / num_clients_);
        int iter_minibatch = 
            ceil(float(max_client_n / num_worker_threads_) / minibatch_size_);
        num_eval_per_client_ = 
            (num_epochs_ * iter_minibatch - 1) 
              / num_eval_minibatch_ + 1;
    }

    // Helper function non-negativise a vector vec 
    inline void RegVec(std::vector<float> & vec, 
            std::vector<float> & vec_result) {
        int len = vec.size();
        for (int i = 0; i < len; i++) {
            vec_result[i] = (vec[i] > 0)? vec[i]: 0;
        }
    }

    // Save results: dicitonary B, coefficients S, loss evaluated on different
    // machines, time between evaluations to disk.
    // Shall be called after calling petuum::PSTableGroup::GlobalBarrier()
    void NMFEngine::SaveResults(int thread_id, petuum::Table<float> & B_table, 
            petuum::Table<float> & loss_table) {
        // size of matrices
        int m = X_matrix_loader_.GetM();
        int client_n = X_matrix_loader_.GetClientN();

        // Caches
        std::vector<float> B_row_cache(m), S_cache(rank_);

        // Output files
        std::ofstream fout_loss, fout_B, fout_S, fout_time;

        // Only thread 0 of client 0 write dictionary B, loss and time to disk
        if (client_id_ == 0 && thread_id == 0) {
            // Write loss to disk
            std::string loss_filename = output_path_ + "/loss.txt";
            fout_loss.open(loss_filename.c_str());
            LOG(INFO) << "Writing loss result to directory: " << output_path_;
            petuum::RowAccessor row_acc;
            std::vector<float> petuum_row_cache(m);
            for (int iter = 0; iter < num_eval_per_client_; ++iter) {
                for (int client = 0; client < num_clients_; ++client) {
                    int row_id = client * num_eval_per_client_ + iter;
                    const auto & row = loss_table.
                        Get<petuum::DenseRow<float> >(row_id, &row_acc);
                    row.CopyToVector(&petuum_row_cache);
                    if (std::abs(petuum_row_cache[0]) > FLT_MIN) {
                        fout_loss << petuum_row_cache[0] << "\t";
                    } else {
                        fout_loss << "N/A" << "\t";
                    }
                }
                fout_loss << "\n";
            }
            fout_loss.close();
            // Write time to disk
            std::string time_filename = output_path_ + "/time.txt";
            fout_time.open(time_filename.c_str());
            for (int iter = 0; iter < num_eval_per_client_; ++iter) {
                for (int client = 0; client < num_clients_; ++client) {
                    int row_id = 
                        (client + num_clients_) * num_eval_per_client_ + iter;
                    const auto & row = loss_table.
                        Get<petuum::DenseRow<float> >(row_id, &row_acc);
                    row.CopyToVector(&petuum_row_cache);
                    if (std::abs(petuum_row_cache[0]) > FLT_MIN) {
                        fout_time << petuum_row_cache[0] << "\t";
                    } else {
                        fout_time << "N/A" << "\t";
                    }
                }
                fout_time << "\n";
            }
            fout_time.close();
            // Write dictionary B to disk
            // with filename output_path_/B.[txt|bin]
            if (output_data_format_ == "text") {
                std::string B_filename = output_path_ + "/B.txt";
                fout_B.open(B_filename.c_str());
            } else if (output_data_format_ == "binary") {
                std::string B_filename = output_path_ + "/B.bin";
                fout_B.open(B_filename.c_str(), std::ios::binary);
            } else {
                LOG(FATAL) << "Unrecognized data format: " << output_data_format_;
            }
            for (int row_id = 0; row_id < rank_; ++row_id) {
                const auto & row = B_table.
                    Get<petuum::DenseRow<float> >(row_id, &row_acc);
                row.CopyToVector(&petuum_row_cache);
                // Non-negativise
                RegVec(petuum_row_cache, B_row_cache);
                for (int col_id = 0; col_id < m; ++col_id) {
                    if (std::abs(B_row_cache[col_id]) < INFINITESIMAL) {
                        B_row_cache[col_id] = 0.0;
                    }
                    if (output_data_format_ == "text") {
                        fout_B << B_row_cache[col_id] << "\t";
                    } else if (output_data_format_ == "binary") {
                        fout_B.write(reinterpret_cast<char*> ( 
                                &(B_row_cache[col_id])), 4);
                    }
                }
                if (output_data_format_ == "text") {
                    fout_B << "\n";
                }
            }
            fout_B.close();
        }
        // Thread 0 of each client save that client's part of S 
        // to output_path_/S.[txt|bin].client_id_
        if (thread_id == 0) {
            if (output_data_format_ == "text") {
                std::string S_filename = output_path_ + "/S.txt." 
                    + std::to_string(client_id_);
                fout_S.open(S_filename.c_str());
            } else if (output_data_format_ == "binary") {
                std::string S_filename = output_path_ + "/S.bin." 
                    + std::to_string(client_id_);
                fout_S.open(S_filename.c_str(), std::ios::binary);
            } else {
                LOG(FATAL) << "Unrecognized data format: " << output_data_format_;
            }
            for (int col_id_client = 0; col_id_client < client_n; 
                    ++col_id_client) {
                if (S_matrix_loader_.GetCol(col_id_client, S_cache)) {
                    for (int row_id = 0; row_id < rank_; ++row_id) {
                        if (std::abs(S_cache[row_id]) < INFINITESIMAL) {
                            S_cache[row_id] = 0.0;
                        }
                        if (output_data_format_ == "text") {
                            fout_S << S_cache[row_id] << "\t";
                        } else if (output_data_format_ == "binary") {
                            fout_S.write(reinterpret_cast<char*> (
                                    &(S_cache[row_id])), 4);
                        }
                    }
                        if (output_data_format_ == "text") {
                            fout_S << "\n";
                        }
                }
            }
            fout_S.close();
        }
    }

    // Init B and S from cache file
    void NMFEngine::LoadCache(int thread_id, petuum::Table<float> & B_table) {
        // size of matrices
        int m = X_matrix_loader_.GetM();
        int client_n = X_matrix_loader_.GetClientN();
        if (client_id_ == 0 && thread_id == 0) {
            // Load B
            std::ifstream fout_B;
            std::vector<float> B_row_cache(m);

            std::string B_filename;
            if (input_data_format_ == "text") {
                B_filename = cache_path_ + "/B.txt";
                fout_B.open(B_filename.c_str());
            } else if (input_data_format_ == "binary") {
                B_filename = cache_path_ + "/B.bin";
                fout_B.open(B_filename.c_str(), std::ios::binary);
            } else {
                LOG(FATAL) << "Unrecognized data format: " << output_data_format_;
            }

            CHECK(fout_B.good()) 
                << "Cache file " << B_filename << " does not exist!";
            for (int row_id = 0; row_id < rank_; ++row_id) {
                petuum::UpdateBatch<float> B_update;
                for (int col_id = 0; col_id < m; ++col_id) {
                    if (input_data_format_ == "text") {
                        fout_B >> B_row_cache[col_id];
                    } else if (input_data_format_ == "binary") {
                        fout_B.read(reinterpret_cast<char*> (
                                &(B_row_cache[col_id])), 4);
                    }
                    B_update.Update(col_id, B_row_cache[col_id]);
                }
                B_table.BatchInc(row_id, B_update);
            }
            fout_B.close();
        }
        // Load S
        if (thread_id == 0) {
            std::ifstream fout_S;
            std::vector<float> S_cache(rank_), 
                S_inc_cache(rank_);

            std::string S_filename;
            if (input_data_format_ == "text") {
                S_filename = cache_path_ + "/S.txt." +
                    std::to_string(client_id_);
                   fout_S.open(S_filename.c_str());
            } else if (input_data_format_ == "binary") {
                S_filename = cache_path_ + "/S.bin." +
                    std::to_string(client_id_);
                   fout_S.open(S_filename.c_str(), std::ios::binary);
            } else {
                LOG(FATAL) << "Unrecognized data format: " << output_data_format_;
            }

            CHECK(fout_S.good()) 
                << "Cache file " << S_filename << " does not exist!";
               for (int col_id_client = 0; col_id_client < client_n; 
                    ++col_id_client) {
                   if (S_matrix_loader_.GetCol(col_id_client, S_cache)) {
                       for (int row_id = 0; row_id < rank_; 
                            ++row_id) {
                        if (input_data_format_ == "text") {
                               fout_S >> S_inc_cache[row_id];
                        } else if (input_data_format_ == "binary") {
                            fout_S.read(reinterpret_cast<char*> (
                                &(S_inc_cache[row_id])), 4);
                        }
                        S_inc_cache[row_id] = 
                            S_inc_cache[row_id] - S_cache[row_id];
                       }
                    S_matrix_loader_.IncCol(col_id_client, S_inc_cache, 0.0);
                   }
                S_matrix_loader_.GetCol(col_id_client, S_cache);
               }
               fout_S.close();
        }
    }

    // Init B table
    void NMFEngine::InitRand(int thread_id, petuum::Table<float> & B_table) {
        if (thread_id != 0 || client_id_ != 0)
            return;
        // size of matrices
        int m = X_matrix_loader_.GetM();
        srand((unsigned)time(NULL));
        std::vector<float> B_row_cache(m);
        // petuum row accessor
        petuum::RowAccessor row_acc;
        for (int row_id = 0; row_id < rank_; ++row_id) {
            petuum::UpdateBatch<float> B_update;
            for (int col_id = 0; col_id < m; ++col_id) {
                B_row_cache[col_id] = double(rand()) / RAND_MAX * 
                    (init_B_high_ - init_B_low_) + init_B_low_;
            }
            for (int col_id = 0; col_id < m; ++col_id) {
                B_update.Update(col_id, B_row_cache[col_id]);
            }
            B_table.BatchInc(row_id, B_update);
        }
    }

    // Stochastic Gradient Descent Optimization
    void NMFEngine::Start() {
        // thread id on a client
        int thread_id = thread_counter_++;
        petuum::PSTableGroup::RegisterThread();
        LOG(INFO) << "client " << client_id_ << ", thread " 
            << thread_id << "registers!";

        // Get dictionary table and loss table
        petuum::Table<float> B_table = 
            petuum::PSTableGroup::GetTableOrDie<float>(0);
        petuum::Table<float> loss_table = 
            petuum::PSTableGroup::GetTableOrDie<float>(1);

        // size of matrices
        int m = X_matrix_loader_.GetM();
        int client_n = X_matrix_loader_.GetClientN();

        // Cache dictionary table 
        Eigen::MatrixXf petuum_table_cache(m, rank_);
        // Accumulate update of dictionary table in minibatch
        Eigen::MatrixXf petuum_update_cache(m, rank_);
        // Cache a column of coefficients S
        Eigen::VectorXf Sj(rank_);
        // Cache a column of update of S_j
        Eigen::VectorXf Sj_inc(rank_);
        // Cache a column of data X 
        Eigen::VectorXf Xj(m);
        Eigen::VectorXf Xj_inc(m);
        // Cache a row of dictionary table
        std::vector<float> petuum_row_cache(m);
    
        // initialize B
        STATS_APP_INIT_BEGIN();
        if (client_id_ == 0 && thread_id == 0) {
            LOG(INFO) << "starting to initialize B";
        }
        if (load_cache_) {// load B and S from cache
            LoadCache(thread_id, B_table);
        } else { // randomly init B to have unit norm
            InitRand(thread_id, B_table);
        }
        if (thread_id == 0 && client_id_ == 0) {
            LOG(INFO) << "matrix B initialization finished!";
        }
        petuum::PSTableGroup::GlobalBarrier();
        STATS_APP_INIT_END();

        // Optimization Loop
        // Timer
        boost::posix_time::ptime beginT = 
            boost::posix_time::microsec_clock::local_time();
        // Step size for optimization
        float step_size_B = init_step_size_B_, step_size_S = init_step_size_S_;

        int num_minibatch = 0;
        for (int iter = 0; iter < num_epochs_; ++iter) {
            // how many minibatches per epoch
            int minibatch_per_epoch = (client_n / num_worker_threads_ > 0)? 
                client_n / num_worker_threads_: 1;
            for (int iter_per_epoch = 0; iter_per_epoch * minibatch_size_ 
                    < minibatch_per_epoch; ++iter_per_epoch) {
                boost::posix_time::time_duration runTime = 
                    boost::posix_time::microsec_clock::local_time() - initT_;
                // Terminate and save states to disk if running time exceeds 
                // limit
                if (maximum_running_time_ > 0.0 && 
                        (float) runTime.total_milliseconds() > 
                        maximum_running_time_*3600*1000) {
                    LOG(INFO) << "Maximum runtime limit activates, "
                        "terminating now!";
                    petuum::PSTableGroup::GlobalBarrier();
                    SaveResults(thread_id, B_table, loss_table);
                    petuum::PSTableGroup::DeregisterThread();
                    return;
                }
                // Update petuum table cache
                petuum::RowAccessor row_acc;
                for (int row_id = 0; row_id < rank_; ++row_id) {
                    const auto & row = 
                        B_table.Get<petuum::DenseRow<float> >(row_id, &row_acc);
                    row.CopyToVector(&petuum_row_cache);
                    for (int col_id = 0; col_id < m; ++col_id) {
                        petuum_table_cache(col_id, row_id) = 
                            petuum_row_cache[col_id];
                    }
                }
                // evaluate obj
                if (num_minibatch % num_eval_minibatch_ == 0) {
                    boost::posix_time::time_duration elapTime = 
                        boost::posix_time::microsec_clock::local_time() - beginT;
                    // evaluate partial obj
                    double obj = 0.0;
                    int num_samples = num_eval_samples_;
                    petuum_table_cache = ( (petuum_table_cache.array() > 0).
                        cast<float>() * petuum_table_cache.array() ).matrix();
                    for (int i = 0; i < num_samples; i++) {
                        int col_id_client = 0;
                        if (S_matrix_loader_.GetRandCol(col_id_client, Sj) 
                                && X_matrix_loader_.GetCol(col_id_client, Xj)) {
                            Xj_inc = Xj - petuum_table_cache * Sj;
                            obj += Xj_inc.squaredNorm();
                        }
                    }
                    obj = obj / num_samples;
                    LOG(INFO) << "iter: " << num_minibatch << ", client " 
                        << client_id_ << ", thread " << thread_id <<
                        " average loss: " << obj;
                    // update loss table
                    loss_table.Inc(client_id_ * num_eval_per_client_ + 
                            num_minibatch / num_eval_minibatch_,  0, 
                            obj/num_worker_threads_);
                    loss_table.Inc((num_clients_+client_id_) * num_eval_per_client_ 
                            + num_minibatch / num_eval_minibatch_, 0, 
                            ((float) elapTime.total_milliseconds()) / 1000 
                            / num_worker_threads_);
                    beginT = boost::posix_time::microsec_clock::local_time();
                }
                step_size_B = init_step_size_B_ * 
                    pow(step_size_offset_B_ + num_minibatch, 
                            -1.0*step_size_pow_B_);
                step_size_S = init_step_size_S_ * 
                    pow(step_size_offset_S_ + num_minibatch, 
                            -1.0*step_size_pow_S_);
                num_minibatch++;
                // clear update table
                petuum_update_cache.fill(0.0);
                // minibatch
                for (int k = 0; k < minibatch_size_; ++k) {
                    int col_id_client = 0;
                    if (S_matrix_loader_.GetRandCol(col_id_client, Sj)
                            && X_matrix_loader_.GetCol(col_id_client, Xj)) {
                        // update S_j
                        for (int iter_S = 0; 
                                iter_S < num_iter_S_per_minibatch_; ++iter_S) {
                            // compute gradient of Sj
                            Sj_inc = step_size_S * (petuum_table_cache.transpose() 
                                    * (Xj - petuum_table_cache * Sj));
                            S_matrix_loader_.IncCol(col_id_client, Sj_inc, 0.0);
                        
                            // get updated S_j
                            S_matrix_loader_.GetCol(col_id_client, Sj);
                        }
                        // update B
                        Xj_inc = Xj - petuum_table_cache * Sj;
                        petuum_update_cache.noalias() += 
                            step_size_B * Xj_inc * Sj.transpose();
                    }
                }
                // Update B_table
                for (int row_id = 0; row_id < rank_; ++row_id) {
                    petuum::UpdateBatch<float> B_update;
                    for (int col_id = 0; col_id < m; ++col_id) {
                        B_update.Update(col_id, 
                                petuum_update_cache(col_id, row_id) 
                                / minibatch_size_);
                    }
                    B_table.BatchInc(row_id, B_update);
                }
                petuum::PSTableGroup::Clock();
                // Update B_table to non-negativise
                std::vector<float> B_row_cache(m);
                for (int row_id = 0; row_id < rank_; ++row_id) {
                    const auto & row = 
                        B_table.Get<petuum::DenseRow<float> >(row_id, &row_acc);
                    row.CopyToVector(&petuum_row_cache);
                    RegVec(petuum_row_cache, B_row_cache);
                    petuum::UpdateBatch<float> B_update;
                    for (int col_id = 0; col_id < m; ++col_id) {
                        B_update.Update(col_id, 
                                (-1.0 * petuum_row_cache[col_id] + 
                                B_row_cache[col_id]) / num_clients_ / 
                                num_worker_threads_);
                    }
                    B_table.BatchInc(row_id, B_update);
                }
                petuum::PSTableGroup::Clock(); 
            }
        }
        // Save results to disk
        petuum::PSTableGroup::GlobalBarrier();
        SaveResults(thread_id, B_table, loss_table);
        petuum::PSTableGroup::DeregisterThread();
    }

    NMFEngine::~NMFEngine() {
    }
} // namespace NMF
