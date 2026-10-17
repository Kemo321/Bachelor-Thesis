#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace dl
{

auto log_error_message(const std::string& message) -> void;

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

}
