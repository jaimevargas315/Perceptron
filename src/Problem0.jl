using Random
using LinearAlgebra # Required for basic matrix operations
using Plots         # For visualization


# Converts the Matrix output from doublemoon (N x 2) to Vector{Vector{Float64}}
function matrix_to_vecvec(X::Matrix{Float64})::Vector{Vector{Float64}}
    return [X[i, :] for i in 1:size(X, 1)]
end

function doublemoon(N::Integer; d::Float64=1.0, r::Float64=10.0, w::Float64=6.0)
    # Pre-allocate vectors to store all generated points and labels
    N_half = N ÷ 2
    X_points = Float64[]
    Y_points = Float64[]
    Labels = Float64[]

    # Define the range for the radius sampling
    R_min = r - w / 2.0
    R_max = r + w / 2.0
    R_range = R_max - R_min

    # --- Moon 1 Generation ---
    for i in 1:N_half
        # Draw R1 from the uniform interval [r - w/2, r + w/2]
        R1 = R_min + R_range * rand()

        # Draw θ1 from the uniform interval [0, π]
        θ1 = π * rand()

        # Form Cartesian X1 and Y1 from the polar R1 and θ1
        X1 = R1 * cos(θ1)
        Y1 = R1 * sin(θ1)

        # Collect the points
        push!(X_points, X1)
        push!(Y_points, Y1)
        push!(Labels, 1.0)
    end

    # --- Moon 2 Generation ---
    for i in 1:N÷2
        # Draw R2 from the uniform interval [r - w/2, r + w/2]
        R2 = R_min + R_range * rand()

        # Draw θ2 from the uniform interval [π, 2π]
        # Range is [π, 2π], total size is π
        θ2 = π + π * rand()

        # Form Cartesian X2 and Y2 from the polar R2 and θ2
        # Offset X2 and Y2 by r and -d respectively (as requested in your original code)
        X2 = R2 * cos(θ2) + r
        Y2 = R2 * sin(θ2) - d

        # Collect the points
        push!(X_points, X2)
        push!(Y_points, Y2)
        push!(Labels, 2.0) # Using 2.0 instead of -1.0 for easier plotting with 'group'
    end

    # Combine the vectors into a single (N x 2) matrix
    X = [X_points Y_points]

    return X, Labels
end

function GaussX(N::Integer; σ²=1.0)

    N_half = N ÷ 2
    σ = sqrt(σ²)

    # --- 1. Generate Class C1 (t=+1): Points forced into Q1 and Q3 ---
    # C1 condition: (x >= 0.0 && y >= 0.0) || (x < 0.0 && y < 0.0)

    R1 = Vector{Float64}[] # R1 holds the collected points for Class 1 (Q1/Q3)
    while length(R1) < N_half
        # Draw candidate (x, y) from a Gaussian distribution centered at zero, scaled by σ.
        x = σ * randn()
        y = σ * randn()

        # NOTE: This Cartesian check replaces the explicit definition of θ₁'s angular range.
        # It forces the point into Quadrants 1 or 3.
        if (x >= 0.0 && y >= 0.0) || (x < 0.0 && y < 0.0)
            push!(R1, [x, y])
        end
    end

    # --- 2. Generate Class C2 (t=-1): Points forced into Q2 and Q4 ---
    # C2 condition: (x < 0.0 && y >= 0.0) || (x >= 0.0 && y < 0.0)

    R2 = Vector{Float64}[] # R2 holds the collected points for Class 2 (Q2/Q4)
    while length(R2) < N_half
        # Draw candidate (x, y) from a Gaussian distribution centered at zero, scaled by σ.
        x = σ * randn()
        y = σ * randn()

        # NOTE: This Cartesian check replaces the explicit definition of θ₂'s angular range.
        # It forces the point into Quadrants 2 or 4.
        if (x < 0.0 && y >= 0.0) || (x >= 0.0 && y < 0.0)
            push!(R2, [x, y])
        end
    end

    # --- 3. Combine Data and Labels ---

    # Concatenate the two sets (R1 and R2)
    X_final = vcat(R1, R2)

    # Assign labels: +1.0 for R1 (first half), -1.0 for R2 (second half)
    label = vcat(ones(N_half), fill(-1.0, N_half))

    return X_final::Vector{Vector{Float64}}, label::Vector{Float64}
end