#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace dl
{

auto log_error_message(const std::string& message) -> void;
auto log_info_message(const std::string& message) -> void;

/**
 * Process-wide asynchronous logger: color stdout at INFO, rotating file at TRACE.
 *
 * Logging is queued onto an spdlog thread pool so host/GPU training loops are not
 * blocked by console or disk I/O. Overflow drops the oldest pending messages.
 */
class Logger
{
public:
    static auto instance() -> Logger&;
    static auto get() -> std::shared_ptr<spdlog::logger>;

    Logger(const Logger&) = delete;
    auto operator=(const Logger&) -> Logger& = delete;
    Logger(Logger&&) = delete;
    auto operator=(Logger&&) -> Logger& = delete;

private:
    Logger();
    ~Logger();

    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace dl

#define LOG_TRACE(...) SPDLOG_LOGGER_TRACE(::dl::Logger::get(), __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(::dl::Logger::get(), __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_INFO(::dl::Logger::get(), __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_WARN(::dl::Logger::get(), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(::dl::Logger::get(), __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(::dl::Logger::get(), __VA_ARGS__)
