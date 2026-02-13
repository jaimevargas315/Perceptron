using LinearAlgebra

function perceptron(𝐱::Vector{Float64}, 𝐰::Vector{Float64})
    if 1+length(𝐱)!=length(𝐰); error("Length of weight vector must be one more than length of data vector"); end
    # Append One: x ← [1 x]T
    x = [1; 𝐱]
    # Matrix multiply: ν = wT x
    𝝂 = 𝐰' * x
    # Explicitly use the signum logic required for Perceptron stability
    if 𝝂 > 1e-9
        y = 1.0
    elseif 𝝂 < -1e-9
        y = -1.0
    else # 𝝂 == 0
        y = 0.0 
    end
    return y::Float64
end
