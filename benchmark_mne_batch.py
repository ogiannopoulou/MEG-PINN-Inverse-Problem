import numpy as np
import meshio
import time
import os
import matplotlib.pyplot as plt

def benchmark_mne_batch(data_file='meg_training_data.npz', 
                        stl_file='MalinBjornsdotterBrain55mm.stl',
                        n_sources=10000, 
                        snr=100.0):
    
    print("=================================================================")
    print("   BATCH BENCHMARK: MNE VS PINN VALIDATION SET")
    print("=================================================================")

    # 1. LOAD DATA & REPLICATE SPLIT
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    data = np.load(data_file)
    X_all = data['X'] # (N, 320, 3)
    Y_all = data['Y'] # (N, 3)
    sensor_positions = data['sensors']
    
    n_total = len(X_all)
    
    # --- EXACT PINN SPLIT LOGIC ---
    # From benchmark_pinns_only_v25.ipynb:
    # indices = np.random.RandomState(42).permutation(n_total)
    # n_train = int(0.8 * n_total)
    # Val indices = indices[n_train:]
    
    print(f"  > Replicating PINN Train/Val Split (Seed 42, 80/20)...")
    indices = np.random.RandomState(42).permutation(n_total)
    n_train = int(0.8 * n_total)
    val_indices = indices[n_train:]
    
    X_val = X_all[val_indices]
    Y_val = Y_all[val_indices]
    
    print(f"  > Total Samples: {n_total}")
    print(f"  > Validation Set: {len(X_val)} samples")
    
    # 2. SETUP SOURCE SPACE & G (COMPUTE ONCE)
    print("\n[One-Time Setup] Computing Lead Field Matrix...")
    start_setup = time.time()
    
    # Source Space Generation
    if os.path.exists(stl_file):
        mesh = meshio.read(stl_file)
        raw_pts = mesh.points * 1e-3 # mm to m
        center_offset = np.mean(raw_pts, axis=0) # Match centering
        mesh_verts = raw_pts - center_offset
        
        # Volumetric Source Space (5 layers)
        layers = [1.0, 0.9, 0.8, 0.7, 0.6]
        n_per_layer = n_sources // len(layers)
        source_layers = []
        for scale in layers:
            idx = np.random.choice(len(mesh_verts), n_per_layer, replace=True)
            source_layers.append(mesh_verts[idx] * scale)
        source_space = np.vstack(source_layers)
    else:
        # Fallback ellipsoidal
        source_space = np.random.randn(n_sources, 3) * 0.05
    
    # Lead Field (Biot-Savart vectorized)
    n_sens = len(sensor_positions)
    n_src = len(source_space)
    
    r_sens = sensor_positions[:, np.newaxis, :]
    r_src = source_space[np.newaxis, :, :]
    R = r_sens - r_src # (n_sens, n_src, 3)
    dist = np.linalg.norm(R, axis=2)
    inv_dist3 = 1.0 / (dist**3 + 1e-15)
    
    mu0 = 4 * np.pi * 1e-7
    factor = mu0 / (4 * np.pi)
    
    # Construct G (n_sens*3, n_src*3) for Vector Fields
    G = np.zeros((n_sens * 3, n_src * 3))
    
    rows_x = np.arange(0, n_sens*3, 3)
    rows_y = np.arange(1, n_sens*3, 3)
    rows_z = np.arange(2, n_sens*3, 3)
    
    # Vectorized Cross Product Logic
    # Src X (cols 0::3): [0, -Rz, Ry]
    G[rows_x, 0::3] = 0.0
    G[rows_y, 0::3] = -R[:,:,2] * factor * inv_dist3
    G[rows_z, 0::3] =  R[:,:,1] * factor * inv_dist3
    
    # Src Y (cols 1::3): [Rz, 0, -Rx]
    G[rows_x, 1::3] =  R[:,:,2] * factor * inv_dist3
    G[rows_y, 1::3] = 0.0
    G[rows_z, 1::3] = -R[:,:,0] * factor * inv_dist3
    
    # Src Z (cols 2::3): [-Ry, Rx, 0]
    G[rows_x, 2::3] = -R[:,:,1] * factor * inv_dist3
    G[rows_y, 2::3] =  R[:,:,0] * factor * inv_dist3
    G[rows_z, 2::3] = 0.0
    
    # Depth Weighting (Compute Once)
    col_norms = np.linalg.norm(G, axis=0)
    col_norms = np.clip(col_norms, 1e-12, None)
    triplet_norms = np.zeros(n_src)
    for i in range(n_src):
        triplet_norms[i] = np.max(col_norms[i*3:i*3+3])
        
    depth_limit = 0.8
    w_diag = (1.0 / triplet_norms) ** depth_limit 
    # Expand to 3N size
    R_diag_vals = np.repeat(w_diag, 3)
    sqrt_R_diag = np.sqrt(R_diag_vals)
    
    # Pre-compute Weighted Gram Matrix for Inversion
    # Cov = (G W) (G W)^T + lambda I
    # We can pre-compute G_weighted
    G_weighted = G * sqrt_R_diag[np.newaxis, :]
    Gram = G_weighted @ G_weighted.T
    
    setup_time = time.time() - start_setup
    print(f"  > Setup Complete ({setup_time:.2f} s)")
    print(f"  > Source Space: {n_src} dipoles")
    print(f"  > Sensor Space: {n_sens} sensors x 3 components")

    # 3. BATCH INVERSION LOOP
    print(f"\n[Batch Processing] Solving for {len(X_val)} validation samples...")
    
    errors = []
    
    # Regularization (SNR based)
    lambda2 = 1.0 / snr**2
    n_measurements = n_sens * 3
    Cov = Gram + lambda2 * np.eye(n_measurements)
    
    # Invert Covariance Matrix ONCE (significant speedup) - O(N_sens^3)
    # Since N_sens (~1000) << N_src (~20000), we act in sensor space
    print("  > Inverting Covariance Matrix...")
    Cov_inv = np.linalg.inv(Cov)
    inverse_operator = G_weighted.T @ Cov_inv # (N_src*3, N_meas)
    
    start_batch = time.time()
    
    for i in range(len(X_val)):
        # Get Measurement vector (flattened)
        B_meas = X_val[i].flatten() 
        true_pos = Y_val[i]
        
        # Add Noise (Optional - to match robustness tests)
        # Using pure data for direct comparison with "Best Possible MNE" performance
        # If PINN validated on noiseless data, MNE should be too.
        # Assuming X_val is noiseless from fenics_2.py
        
        # Apply Inverse Operator (Matrix-Vector Mult) - O(N_meas * N_src)
        # w = Cov_inv @ B_meas
        # J_est_weighted = G_weighted.T @ w
        # Combined: J_est_weighted = inverse_operator @ B_meas
        
        J_est_weighted = inverse_operator @ B_meas
        
        # Unweight
        J_est = J_est_weighted * sqrt_R_diag
        
        # Localize
        J_vectors = J_est.reshape(n_src, 3)
        J_magnitudes = np.linalg.norm(J_vectors, axis=1)
        peak_idx = np.argmax(J_magnitudes)
        est_pos = source_space[peak_idx]
        
        # Error (cm)
        err = np.linalg.norm(est_pos - true_pos) * 100
        errors.append(err)
        
        if i % 100 == 0:
            print(f"    Sample {i}/{len(X_val)}: Error = {err:.2f} cm")
            
    batch_time = time.time() - start_batch
    errors = np.array(errors)
    
    # 4. REPORT STATISTICS
    print("\n=================================================================")
    print("   COMPARISON METRICS (MNE vs PINN)")
    print("=================================================================")
    print(f"Validation Samples: {len(errors)}")
    print(f"Inverse Method:     MNE (Depth Weighted, Vector Field)")
    print(f"Source Space:       {n_src} Volumetric Sources")
    print(f"Batch Time:         {batch_time:.2f} s ({batch_time/len(errors)*1000:.1f} ms/sample)")
    print("-----------------------------------------------------------------")
    print(f"MEAN ERROR:         {np.mean(errors):.4f} cm")
    print(f"MEDIAN ERROR:       {np.median(errors):.4f} cm")
    print(f"STD DEV:            {np.std(errors):.4f} cm")
    print(f"MIN ERROR:          {np.min(errors):.4f} cm")
    print(f"MAX ERROR:          {np.max(errors):.4f} cm")
    print("-----------------------------------------------------------------")
    
    # Save for plotting
    np.savetxt("mne_validation_errors.txt", errors)
    
    # Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f} cm')
    plt.axvline(np.median(errors), color='green', linestyle='-', label=f'Median: {np.median(errors):.2f} cm')
    plt.title(f"MNE Localization Error Distribution (N={len(errors)})")
    plt.xlabel("Localization Error (cm)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mne_validation_histogram.png")
    print("Histogram saved to 'mne_validation_histogram.png'")

if __name__ == "__main__":
    benchmark_mne_batch()
