#include "DeepLearnLib/Logger.hpp"

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <exception>
#include <filesystem>
#include <vector>

namespace dl
{
namespace
{
    constexpr std::size_t kAsyncQueueSize = 8192;
    constexpr std::size_t kAsyncWorkerThreads = 1;
    constexpr std::size_t kRotatingFileBytes = 5 * 1024 * 1024;
    constexpr std::size_t kRotatingFileCount = 3;
} // namespace

// Asynchronous logger: console from info, rotating file from trace.
// The file is optional, so a missing logs directory does not silence stdout.
Logger::Logger()
{
    spdlog::init_thread_pool(kAsyncQueueSize, kAsyncWorkerThreads);

    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    stdout_sink->set_level(spdlog::level::info);
    stdout_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    std::vector<spdlog::sink_ptr> sinks { stdout_sink };
    try
    {
        std::filesystem::create_directories("logs");
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            "logs/framework.log", kRotatingFileBytes, kRotatingFileCount);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
        sinks.push_back(std::move(file_sink));
    }
    catch (const std::exception&)
    {
        // Keep stdout only when the rotating file cannot be created.
    }

    // Queue overflow blocks the caller instead of dropping records.
    logger_ = std::make_shared<spdlog::async_logger>("deeplearn", sinks.begin(), sinks.end(), spdlog::thread_pool(),
        spdlog::async_overflow_policy::block);
    logger_->set_level(spdlog::level::trace);
    // A record at info or above flushes the logger; trace and debug stay buffered until then.
    logger_->flush_on(spdlog::level::info);

    spdlog::register_logger(logger_);
    spdlog::set_default_logger(logger_);
}

// Drains the queue and shuts spdlog down, so the last records are not lost with the process.
Logger::~Logger()
{
    if (logger_)
    {
        // Drain the async queue before shutdown, or the last records disappear with the process.
        logger_->flush();
        logger_.reset();
    }
    spdlog::shutdown();
}

// One instance per process. The static object lives until the program ends.
auto Logger::instance() -> Logger&
{
    static Logger logger;
    return logger;
}

// Shared spdlog handle for the log_* functions and the macros.
auto Logger::get() -> std::shared_ptr<spdlog::logger>
{
    return instance().logger_;
}

// An error-level record on the default logger.
auto log_error_message(const std::string& message) -> void
{
    Logger::get()->error("{}", message);
}

// An info-level record on the default logger.
auto log_info_message(const std::string& message) -> void
{
    Logger::get()->info("{}", message);
}

// A debug-level record on the default logger.
auto log_debug_message(const std::string& message) -> void
{
    Logger::get()->debug("{}", message);
}

// Pushes the buffer out to the sinks. Needed before something reads the log file from disk.
auto log_flush() -> void
{
    Logger::get()->flush();
}

} // namespace dl
