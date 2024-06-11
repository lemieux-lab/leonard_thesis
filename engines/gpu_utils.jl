using CUDA
using CUDA.NVML: compute_processes, index, uuid, name

"""
    CUDA.device!()

Selects the first GPU with no process associated yet. Do nothing if none exist.
"""
function CUDA.device!()
    for dev in CUDA.NVML.devices()
        if compute_processes(dev) === nothing
            i = index(dev)
            println("Switching to GPU-$i: $(name(dev)) (uuid: $(uuid(dev)))")
            CUDA.device!(i)
            break
        end
    end
end