using LinearAlgebra
using Plots
using Random

## Problem 8: Newton’s Method on Himmelblau’s Cost Function

# --- 1. Himmelblau Cost Function (J(w)) ---

function himmelblau_cost(w::Vector{Float64})::Float64
    """ J(w) = (w₁² + w₂ - 11)² + (w₁ + w₂² - 7)² """
    w1, w2 = w[1], w[2]
    term1 = w1^2 + w2 - 11.0
    term2 = w1 + w2^2 - 7.0
    return term1^2 + term2^2
end

# --- 2. Himmelblau Gradient Function (g = ∇J(w)) ---

function himmelblau_gradient(w::Vector{Float64})::Vector{Float64}
    """ Gradient of Himmelblau's function. """
    w1, w2 = w[1], w[2]
    T1 = w1^2 + w2 - 11.0
    T2 = w1 + w2^2 - 7.0
    
    dw1 = 4.0 * w1 * T1 + 2.0 * T2
    dw2 = 2.0 * T1 + 4.0 * w2 * T2
    
    return [dw1, dw2]
end

# --- 3. Himmelblau Hessian Function (H) ---

function himmelblau_hessian(w::Vector{Float64})::Matrix{Float64}
    """ Hessian matrix of Himmelblau's function. """
    w1, w2 = w[1], w[2]
    
    # Pre-calculate terms
    T1 = w1^2 + w2 - 11.0
    T2 = w1 + w2^2 - 7.0
    
    # H11: 4*T1 + 8*w1^2 + 2
    H11 = 4.0 * T1 + 8.0 * w1^2 + 2.0
    
    # H22: 2 + 4*T2 + 8*w2^2
    H22 = 2.0 + 4.0 * T2 + 8.0 * w2^2
    
    # H12 = H21: 4*w1 + 4*w2
    H12 = 4.0 * w1 + 4.0 * w2
    
    return [
        H11 H12; 
        H12 H22
    ]
end

# --- 4. Newton's Method Algorithm ---

function newtons_method_himmelblau(
    initial_w::Vector{Float64};
    η::Float64 = 1.0,
    max_iterations::Integer = 100,
    tolerance::Float64 = 1e-6
)::Tuple{Vector{Vector{Float64}}, Bool}
    """
    Performs Newton's Method: w ← w - η * H⁻¹ * g.
    Returns: (weight_history, converged_successfully)
    """
    
    w = copy(initial_w)
    weight_history = Vector{Float64}[]
    push!(weight_history, copy(w))
    converged = false
    
    for k in 1:max_iterations
        g = himmelblau_gradient(w)
        H = himmelblau_hessian(w)
        
        # Check for convergence based on gradient norm
        if norm(g) < tolerance
            # println("Converged by small gradient after $k steps.")
            converged = true
            break
        end

        try
            # Calculate Newton Step: Δw = -H⁻¹ * g
            H_inv = inv(H)
            delta_w = -H_inv * g
            
            # Check for stagnation or divergence (large step)
            if norm(delta_w) > 1000.0 # Heuristic to detect explosion/divergence
                 # println("Run terminated due to step explosion.")
                 break
            end

            # Update rule: w ← w + η * Δw
            w_new = w + η * delta_w
            
            # Check for weight change convergence
            if norm(w_new - w) < tolerance
                w = w_new
                push!(weight_history, copy(w))
                converged = true
                break
            end
            
            w = w_new
            push!(weight_history, copy(w))

        catch e
            # Catches exceptions like "SingularException" (Hessian is non-invertible)
            # println("Run terminated due to singular Hessian matrix at step $k.")
            break
        end
    end
    
    return weight_history, converged
end


# --- 5. Plotting Function (50 Trajectories) ---

function plot_himmelblau_50_trajectories_newton(
    all_w_histories::Vector{Vector{Vector{Float64}}}, 
    final_results::Vector{Tuple{Vector{Float64}, Bool}}
)
    """ Plots the Himmelblau contour and all 50 Newton's Method trajectories. """
    
    w_stars = [
        [3.0, 2.0],        
        [-2.805118, 3.131312], 
        [-3.779310, -3.283186], 
        [3.584428, -1.848126]  
    ]
    
    range_lim = 6.0
    w_range = range(-range_lim, stop=range_lim, length=100)
    J_surface = [himmelblau_cost([w1, w2]) for w2 in w_range, w1 in w_range]

    p = contour(w_range, w_range, J_surface, 
                fill=true, 
                levels=vcat(range(0.1, 10.0, length=10), range(20.0, 500.0, length=10)), 
                cbar=false, 
                title="Newton's Method (50 Runs, η=1.0, Gaussian σ=1)",
                xlabel="w₁", 
                ylabel="w₂",
                legend=:bottomright,
                colormap=:viridis,
                size=(800, 800), # Explicitly setting a square plot size
                aspect_ratio=:equal, # Ensure axes are scaled equally
                xlims=(-range_lim, range_lim), # NEW: Explicitly set x limits
                ylims=(-range_lim, range_lim)) # NEW: Explicitly set y limits

    scatter!(p, [w[1] for w in w_stars], [w[2] for w in w_stars], 
             label="Global Minima (J=0)", 
             marker=:star, markersize=10, markercolor=:red, markerstrokecolor=:black)
             
    converged_points = Vector{Float64}[]
    failed_points = Vector{Float64}[]

    for (i, w_history) in enumerate(all_w_histories)
        if isempty(w_history) || length(w_history) < 2
            continue
        end
        
        w1_path = [w[1] for w in w_history]
        w2_path = [w[2] for w in w_history]
        
        # Plot the trajectory path
        plot!(p, w1_path, w2_path, 
              label=nothing, 
              color=:lime, 
              linewidth=0.5, 
              alpha=0.5)

        # Separate final points based on success
        if final_results[i][2]
            push!(converged_points, w_history[end])
        else
            push!(failed_points, w_history[end])
        end
    end
    
    # Plot successfully converged final points
    scatter!(p, [w[1] for w in converged_points], [w[2] for w in converged_points], 
             label="Converged (Low J)", 
             markersize=4, markershape=:circle, markercolor=:yellow, markerstrokecolor=:black, alpha=0.8)

    # Plot failed/diverged final points (where the run stopped)
    scatter!(p, [w[1] for w in failed_points], [w[2] for w in failed_points], 
             label="Failed/Diverged", 
             markersize=4, markershape=:xcross, markercolor=:magenta, alpha=0.8)

    display(p)
end

# --- 6. Execution (50 Runs) ---
const NUM_RUNS = 50
eta_val = 1.0 
iterations = 100

Random.seed!(42) 

all_w_histories = Vector{Vector{Float64}}[]
final_results = Vector{Tuple{Vector{Float64}, Bool}}(undef, NUM_RUNS)

println("--- Starting 50 Newton's Method Runs on Himmelblau (η=1.0) ---")

for i in 1:NUM_RUNS
    # Gaussian initialization: μ=0, σ=1
    initial_w = randn(2) 
    
    (w_history, converged) = newtons_method_himmelblau(
        initial_w; 
        η=eta_val, 
        max_iterations=iterations
    )
    
    push!(all_w_histories, w_history)
    final_w = isempty(w_history) ? initial_w : w_history[end]
    final_results[i] = (final_w, converged)
end

# Generate Plot showing all 50 trajectories
plot_himmelblau_50_trajectories_newton(all_w_histories, final_results)

# --- Discussion Metrics ---
total_converged = sum(r[2] for r in final_results)
println("\n--- Newton's Method on Himmelblau Results (50 Runs) ---")
println("Total Runs Converged Successfully (J ≈ 0): $total_converged / $NUM_RUNS")
# Select a final point and check its cost
example_final_cost = himmelblau_cost(final_results[1][1])
println("Example Final Cost (Run 1): J(w_final) = $(round(example_final_cost, digits=6))")