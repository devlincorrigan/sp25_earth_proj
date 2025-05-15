def finite_vol(Delta_x, Delta_y, Delta_t, g, U, V, H, S_u, S_v):

# --- Lax-Friedrichs Step ---

    HU = H*U # (HU)_{ij}^n = third component of \vec f_{ij}^{n, Q}
    HV = H*V # (HV)_{ij}^n = third component of \vec g_{ij}^{n, Q}

    # predict H_{i+1/2, j}^{n+1/2}
    H_mid_xt = 0.5 * (H[1:, :] + H[0:-1, :]) - (0.5 * Delta_t / Delta_x) * (HU[1:,:] - HU[0:-1,:])

    # predict H_{i, j+1/2}^{n+1/2}
    H_mid_yt = 0.5 * (H[:, 1:] + H[:, 0:-1]) - (0.5 * Delta_t / Delta_y) * (HV[:, 1:] - HV[:, 0:-1])

    # (HU^2 + gH^2 / 2)_{ij}^n = first component of \vec f_{ij}^{n, Q}
    f_u = HU * U + g * (H**2) / 2 

    # predict (HU)_{i+1/2, j}^{n+1/2}
    HU_mid_xt = 0.5 * (HU[1:, :] + HU[0:-1, :]) - (0.5 * Delta_t / Delta_x) * (f_u[1:, :] - f_u[0:-1, :])
    
    # (HUV)_{ij}^n = first component of \vec g_{ij}^{n, Q}
    g_u = HU * V 

    # predict (HU)_{i, j+1/2}^{n+1/2}
    HU_mid_yt = 0.5 * (HU[:, 1:] + HU[:, 0:-1]) - (0.5 * Delta_t / Delta_y) * (g_u[:, 1:] - g_u[:, 0:-1])

    # (HUV)_{ij}^n = second component of \vec f_{ij}^{n, Q}
    f_v = g_u 

    # predict (HV)_{i+1/2, j}^{n+1/2} 
    HV_mid_xt = 0.5 * (HV[1:, :] + HV[0:-1, :]) - (0.5 * Delta_t / Delta_x) * (f_v[1:, :] - f_v[0:-1, :])

    # (HV^2 + gH^2 / 2)_{ij}^n = second component of \vec g_{ij}^n, Q}
    g_v = HV * V + g * (H**2) / 2 

    #predict (HV)_{i, j+1/2}^{n+1/2}
    HV_mid_yt = 0.5 * (HV[:, 1:] + HV[:, 0:-1]) - (0.5 * Delta_t / Delta_y) * (g_v[:, 1:] - g_v[:, 0:-1])

# --- compute the fluxes at t^{n+1/2} ---

    # [(HU^2) / H + gH^2 / 2]_{i+1/2, j}^{n+1/2} = first component of \vec f_{i+1/2, j}^{n+1/2, Q}
    f_mid_xt_u = HU_mid_xt * HU_mid_xt / H_mid_xt + g * (H_mid_xt**2) / 2

    # [(HU)(HV) / H]_{i, j+1/2}^{n+1/2} = first component of \vec g_{i, j+1/2}^{n+1/2, Q}
    g_mid_yt_u = HU_mid_yt * HV_mid_yt / H_mid_yt

    f_mid_xt_v = HU_mid_xt * HV_mid_xt / H_mid_xt
    g_mid_yt_v = HV_mid_yt * HV_mid_yt / H_mid_yt + g * (H_mid_yt**2) / 2

# --- Finite Volume Step ---
    # third component of Q_new
    H_new = H[1:-1,1:-1] - (Delta_t/Delta_x)*(HU_mid_xt[1:,1:-1]-HU_mid_xt[0:-1,1:-1]) - (Delta_t/Delta_y)*(HV_mid_yt[1:-1,1:]-HV_mid_yt[1:-1,0:-1])
    
    # first component of Q_new
    HU_new = HU[1:-1,1:-1] - (Delta_t / Delta_x) * (f_mid_xt_u[1:, 1:-1] - f_mid_xt_u[0:-1, 1:-1]) - (Delta_t / Delta_y) * (g_mid_yt_u[1:-1, 1:] - g_mid_yt_u[1:-1, 0:-1]) + Delta_t * S_u * 0.5 * (H[1:-1,1:-1] + H_new)

    #second component of Q_new
    HV_new = HV[1:-1, 1:-1] - (Delta_t / Delta_x) * (f_mid_xt_v[1:, 1:-1] - f_mid_xt_v[0:-1, 1:-1]) - (Delta_t / Delta_y) * (g_mid_yt_v[1:-1, 1:] - g_mid_yt_v[1:-1, 0:-1]) + Delta_t * S_v * 0.5 * (H[1:-1,1:-1]+ H_new)
    
    U_new = HU_new / H_new
    V_new = HV_new / H_new
    return (U_new, V_new, H_new)