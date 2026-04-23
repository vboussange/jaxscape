using Circuitscape
using Statistics

config_path = ARGS[1]
repeats = parse(Int, ARGS[2])
output_path = ARGS[3]

Circuitscape.compute(config_path)

timings = Float64[]
for _ in 1:repeats
    elapsed = @elapsed Circuitscape.compute(config_path)
    push!(timings, elapsed)
end

timings_json = join(string.(timings), ", ")
payload = "{\"status\":\"ok\",\"timings_seconds\":[$timings_json],\"median_seconds\":$(median(timings)),\"note\":\"\"}"

open(output_path, "w") do io
    write(io, payload)
end
