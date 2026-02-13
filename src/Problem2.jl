function trainPerceptron(X::Vector{Vector{Float64}}, 𝐝::Vector{Float64}, η::Float64; 
                         𝐰=nothing, maxIter::Integer=1, tol=1e-3) # maxIter set to 1
    
    # Initialization (Must be here per template)
    if 𝐰===nothing; 𝐰 = randn(length(X[1])+1); end
    N = length(X)
    shuffleIndex = collect(1:N)
    
    # We remove the unused 'e = zeros(maxIter)' from your original template to avoid type issues.
    # We also remove the convergence check and the outer loop control logic as we will manage
    # convergence and epoch control externally for the 30-trial loop.
    
    # We shuffle the data once per call (once per epoch)
    shuffle!(shuffleIndex)
    X_shuffled = X[shuffleIndex]
    D_shuffled = 𝐝[shuffleIndex]
    
    # Core SGD Loop (Inner loop body from your template)
    for n ∈ 1:N
        # compute the response of the perceptron
        y = perceptron(X_shuffled[n], 𝐰) # Use shuffled data
        
        # compute the error (target minus actual output)
        e = D_shuffled[n] - y      
        
        # If the error is non-zero, update the weights
        if abs(e) > 1e-9 # Check if e is effectively non-zero
            
            # Augment the input vector x_n for the update rule
            x_aug = [1.0; X_shuffled[n]] # Use shuffled data
            
            # Update the weights: w <- w + eta * e * x_aug
            𝐰 .+= η * e * x_aug 
        end
    end

    # Return updated weights and the epoch count (which is always 1)
    return 𝐰::Vector{Float64}
end