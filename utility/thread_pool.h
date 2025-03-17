#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <atomic>

using namespace std;

class ThreadPool
{
public:
    // // Constructor to creates a thread pool with given
    // number of threads
    // TODO: fix argument flexibility
    ThreadPool(size_t num_threads = thread::hardware_concurrency(), bool see_progress = true)
    {
        if (num_threads == 0)
        {
            printf("Number of thread is 0, aborting\n");
            abort();
        }
        size_t worker_thread;
        if (see_progress)
        {
            worker_thread = num_threads - 1;
            progress_thread_ = thread([this]()
                                      { display_progress(); });
        }
        else
        {
            worker_thread = num_threads;
        }
        // Creating worker threads
        for (size_t i = 0; i < worker_thread; ++i)
        {
            worker_thread_.emplace_back([this]
                                        {
                    while (true) {
                        function<void()> task;
                        // The reason for putting the below code
                        // here is to unlock the queue before
                        // executing the task so that other
                        // threads can perform enqueue tasks
                        {
                            // Locking the queue so that data
                            // can be shared safely
                            unique_lock<mutex> lock(
                                queue_mutex_);
    
                            // Waiting until there is a task to
                            // execute or the pool is stopped
                            cv_.wait(lock, [this] {
                                return !tasks_.empty() || stop_;
                            });
    
                            // exit the thread in case the pool
                            // is stopped and there are no tasks
                            if (stop_ && tasks_.empty()) {
                                return;
                            }
    
                            // Get the next task from the queue
                            task = move(tasks_.front());
                            tasks_.pop();
                        }
                        active_threads++;
                        task();
                        active_threads--;
                        cv_signal_.notify_one();
                    } });
        }
    };

    // Destructor to stop the thread pool
    ~ThreadPool()
    {
        threads_on = false;
        {
            // Lock the queue to update the stop flag safely
            unique_lock<mutex> lock(queue_mutex_);
            stop_ = true;
        }

        // Notify all threads
        cv_.notify_all();

        // Joining all worker threads to ensure they have
        // completed their tasks
        for (auto &thread : worker_thread_)
        {
            thread.join();
        }
        bool tmp = progress_thread_.joinable();
        progress_thread_.join();
        std::cout << "Thread pool: destructor called" << std::endl;
    };

    // Enqueue task for execution by the thread pool
    void enqueue(function<void()> task)
    {
        {
            unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(move(task));
        }
        cv_.notify_one();
    };

    // Monitor progress
    void display_progress()
    {
        while (threads_on)
        {
            {
                // std::unique_lock<std::mutex> lock(stats_mutex_);
                std::cout << "task remaining: " << tasks_.size() << " active threads: " << active_threads << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    };

    void wait_all()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_signal_.wait(lock, [this]()
                     { return tasks_.empty() && active_threads == 0; });
            std::cout << "WAIT_ALL: task remaining: " << tasks_.size() << " active threads: " << active_threads << std::endl;
        }
    }

private:
    // Vector to store worker threads
    vector<thread> worker_thread_;
    // Monitor thread
    thread progress_thread_;

    // Queue of tasks
    queue<function<void()>> tasks_;

    // Mutex to synchronize access to shared data
    mutex queue_mutex_;
    
    std::atomic<bool> threads_on = true;

    // Condition variable to signal changes in the state of
    // the tasks queue
    condition_variable cv_;

    condition_variable cv_signal_;

    // Flag to indicate whether the thread pool should stop
    // or not
    bool stop_ = false;

    std::atomic<int> active_threads = 0;
};

#endif // THREAD_POOL_H_