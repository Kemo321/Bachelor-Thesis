# Logger.cpp

Builds one asynchronous spdlog logger: a color stdout sink and an optional rotating file. `log_to_file` can add a third sink for the current run.

## Logger::Logger

Creates a thread pool (queue of 8192, one worker) and the logger `deeplearn`. Stdout accepts levels from info, with a time and level pattern. The file `logs/framework.log` rotates at 5 MiB, keeps 3 files, and records from trace together with the source location. When the directory or the file cannot be created, only stdout remains. The logger is the default, its level is trace, and it flushes from info upward. Queue overflow blocks the sender.

## Logger::~Logger

If the handle is still alive, it drains the queue and releases it, then shuts spdlog down. Otherwise the end of the process drops records still sitting in the queue.

## Logger::instance

Returns the only instance. The function-local static object is created on the first call and lives until the program ends.

## Logger::get

Returns a `shared_ptr` to the logger from `instance()`. The `log_*` functions and the `LOG_*` macros use that handle.

## log_error_message

Passes the text to the logger at error level.

## log_info_message

Passes the text to the logger at info level.

## log_to_file

Creates the parent directory and adds a file sink at info level. The file is truncated, so it holds the current run. The pattern is time, level, and message, without the source location used by `logs/framework.log`. Queue records already buffered are flushed before the sink is attached.

## log_debug_message

Passes the text to the logger at debug level.

## log_flush

Pushes the logger buffer out to the sinks. Needed before something reads the log file from disk.
