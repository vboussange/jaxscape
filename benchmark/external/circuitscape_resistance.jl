using Circuitscape
using Statistics

config_path = ARGS[1]
repeats = parse(Int, ARGS[2])
output_path = ARGS[3]

function escape_json_string(value::String)
    replace(value, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "")
end

function write_payload(status::String, timings::Vector{Float64}, note::String)
    timings_json = isempty(timings) ? "" : join(string.(timings), ", ")
    median_json = isempty(timings) ? "null" : string(median(timings))
    payload = "{\"status\":\"$(escape_json_string(status))\",\"timings_seconds\":[$(timings_json)],\"median_seconds\":$(median_json),\"note\":\"$(escape_json_string(note))\"}"

    open(output_path, "w") do io
        write(io, payload)
    end
end

try
    Circuitscape.compute(config_path)

    timings = Float64[]
    for _ in 1:repeats
        elapsed = @elapsed Circuitscape.compute(config_path)
        push!(timings, elapsed)
    end

    write_payload("ok", timings, "")
catch err
    write_payload("failed", Float64[], sprint(showerror, err))
end
