using LinearAlgebra

function trainBatchPerceptron(X::Vector{Vector{Float64}}, 𝐝::Vector{Float64}, η::Float64; 𝐰=nothing, maxIter::Integer=50, tol=1e-9)
    if 𝐰===nothing; 𝐰 = randn(length(X[1])+1); end
    iter = 0
    N = length(X)
    for outer iter ∈ 1:maxIter
        𝐰_old = copy(𝐰) #save centers to check for convergence

        # Initialize the total correction vector for this batch (epoch)
        # This vector accumulates sum_{n | x_n ∈ M} e_n * x_n
        Δ𝐰_batch = zeros(length(𝐰))
        
        # Loop through all training examples (the batch)
        for n ∈ 1:N
            
            # Augment x_n (the input vector) for calculations
            x_aug = [1.0; X[n]]
            
            # Compute the weighted sum and activation output
            nu = dot(𝐰, x_aug)
            y = sign(nu)
            
            # Compute the error: e = desired - actual
            e = 𝐝[n] - y
            
            # If misclassified (error is non-zero)
            if abs(e) > 1e-12 
                # Accumulate the correction term (e_n * x_n)
                # This performs: Δ𝐰_batch += e * x_aug
                Δ𝐰_batch .+= e * x_aug 
            end
        end
        
        # APPLY THE BATCH UPDATE: w <- w + η * Δ𝐰_batch
        # This step is performed ONLY ONCE after the entire dataset (batch) is processed
        𝐰 .+= η * Δ𝐰_batch

        if norm(𝐰-𝐰_old) < tol #check for convergence
            break
        end
    end
    return 𝐰::Vector{Float64}, iter::Integer
end
