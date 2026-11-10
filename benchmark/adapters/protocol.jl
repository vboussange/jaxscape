# What a Julia adapter shares with the runner: the protocol of `protocol.py`.
using JSON, NPZ

"""Return the cell file named on the command line, and the arrays of its case."""
function read_cell()
    cell = JSON.parsefile(ARGS[1])
    names = filter(name -> endswith(name, ".npy"), readdir(cell["case"]))
    arrays = Dict(first(splitext(name)) => npzread(joinpath(cell["case"], name)) for name in names)
    return cell, arrays
end

"""Time `call` as `time_calls` of `protocol.py` does, and return the output of
its last call, the warm-up time and the times of the timed calls, in seconds."""
function time_calls(cell, call)
    function timed()
        touch(cell["progress"])
        start = time_ns()
        output = call()
        return output, (time_ns() - start) / 1e9
    end
    output, warmup = timed()
    repeats = warmup > cell["single_call_after"] ? min(cell["repeats"], 1) : cell["repeats"]
    times = Float64[]
    for _ in 1:repeats
        output, elapsed = timed()
        push!(times, elapsed)
    end
    return output, warmup, times
end

function write_result(cell, output, warmup, times; version, files = false)
    npzwrite(cell["output"], output)
    result = Dict("version" => version, "files" => files, "warmup_s" => warmup, "times_s" => times)
    write(cell["result"], JSON.json(result))
end
