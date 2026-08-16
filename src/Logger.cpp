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
        // Keep the color stdout sink if the rotating file cannot be created.
    }

    logger_ = std::make_shared<spdlog::async_logger>("deeplearn", sinks.begin(), sinks.end(), spdlog::thread_pool(),
        spdlog::async_overflow_policy::overrun_oldest);
    logger_->set_level(spdlog::level::trace);
    logger_->flush_on(spdlog::level::err);

    spdlog::register_logger(logger_);
    spdlog::set_default_logger(logger_);
}

Logger::~Logger()
{
    if (logger_)
    {
        logger_->flush();
        logger_.reset();
    }
    spdlog::shutdown();
}

auto Logger::instance() -> Logger&
{
    static Logger logger;
    return logger;
}

auto Logger::get() -> std::shared_ptr<spdlog::logger>
{
    return instance().logger_;
}

auto log_error_message(const std::string& message) -> void
{
    Logger::get()->error("{}", message);
}

auto log_info_message(const std::string& message) -> void
{
    Logger::get()->info("{}", message);
}

} // namespace dl
