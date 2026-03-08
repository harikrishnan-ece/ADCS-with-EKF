import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

# ----------------------------------------
# Utility Functions
# ----------------------------------------

def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def quat_multiply(q, r):
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_rot(q):
    q = normalize(q)
    q0,q1,q2,q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1**2+q2**2)]
    ])

def attitude_error(R_true, R_est):
    val = (np.trace(R_true.T @ R_est) - 1)/2
    val = np.clip(val, -1, 1)
    return np.rad2deg(np.arccos(val))

# ----------------------------------------
# Simulation Parameters
# ----------------------------------------

dt = 0.05
time = np.arange(0, 60, dt)

I = np.diag([10, 8, 6])
I_inv = np.linalg.inv(I)

true_q = np.array([1.0,0,0,0])
est_q  = np.array([1.0,0,0,0])

omega = np.array([0.01, -0.02, 0.015])

gyro_bias = np.array([0.001, -0.001, 0.0005])
est_bias = np.zeros(3)

P = np.eye(6)*0.01
Q = np.eye(6)*1e-6

# Stable control gains
Kp = 8
Kd = 4
max_rw_torque=0.02
I_rw=0.001
rw_speed=0.0

error_history = []
omega_history = []
torque_history = []
bias_history = []
rw_speed_history = []
disturbance_history = []


# ----------------------------------------
# MAIN LOOP
# ----------------------------------------

for t in time:
   # --- Quaternion Error ---
    q_error = quat_multiply(quat_conjugate(est_q), true_q)
    q_error = normalize(q_error)
        # Desired control torque from PD controller
    if t < 40:
        torque_cmd = -Kp*q_error[1:] - Kd * omega
    else:
        Kd_safe = 2
        torque_cmd = -Kd_safe*omega

    # Reaction wheel saturation
    torque_rw = np.clip(torque_cmd, -max_rw_torque, max_rw_torque)

    # Wheel dynamics
    rw_speed_dot = torque_rw / I_rw
    rw_speed += rw_speed_dot * dt

    # Satellite gets opposite torque
    torque = -torque_rw
        

    # Saturation
    max_torque = 0.05
    torque = np.clip(torque, -max_torque, max_torque)
        # Small random disturbance torque
    disturbance = np.random.normal(0, 0.0001, 3)

    # Add disturbance to control torque
    omega_dot = I_inv @ (torque + disturbance - np.cross(omega, I @ omega))
    omega += omega_dot * dt
    omega_quat = np.concatenate([[0], omega])
    q_dot = 0.5 * quat_multiply(true_q, omega_quat)
    true_q = normalize(true_q + q_dot*dt)

    # --- Gyro Measurement ---
    gyro_meas = omega + gyro_bias + np.random.normal(0, 0.0005, 3)

    # Remove estimated bias
    gyro_corrected = gyro_meas - est_bias
    omega_quat_meas = np.concatenate([[0], gyro_corrected])
    q_dot_est = 0.5 * quat_multiply(est_q, omega_quat_meas)
    est_q = normalize(est_q + q_dot_est*dt)

    # ---- Improved EKF Covariance Propagation ----

    F = np.eye(6)

    # Skew-symmetric matrix of angular velocity
    wx, wy, wz = omega

    Omega_skew = np.array([
        [ 0,   -wz,  wy],
        [ wz,   0,  -wx],
        [-wy,  wx,   0 ]
    ])

    # Attitude error dynamics
    F[0:3,0:3] -= Omega_skew * dt
    # Bias coupling
    F[0:3,3:6] = -np.eye(3) * dt
    # Bias random walk (identity)
    F[3:6,3:6] = np.eye(3)
    P = F @ P @ F.T + Q
    # --- Star Tracker Measurement (Failure after 40s) ---
    if t < 40:
        # Simulated star tracker measurement
        meas_q = normalize(true_q + np.random.normal(0, 0.01, 4))
        # Quaternion innovation
        q_tilde = quat_multiply(meas_q, quat_conjugate(est_q))
        q_tilde = normalize(q_tilde)
        innovation = q_tilde[1:]

        # Measurement matrix
        H = np.zeros((3,6))
        H[:,0:3] = np.eye(3)
        R_meas = np.eye(3) * 1e-3
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R_meas)
        delta_x = K @ innovation

        # Small angle correction
        delta_q = np.concatenate([[1], 0.5*delta_x[0:3]])
        delta_q = normalize(delta_q)
        est_q = normalize(quat_multiply(est_q, delta_q))

        # Update bias
        est_bias = est_bias + delta_x[3:6]
        P = (np.eye(6) - K @ H) @ P
    

    # --- Error Tracking ---
    R_est = quat_to_rot(est_q)
    R_true = quat_to_rot(true_q)
    err = attitude_error(R_true, R_est)
    error_history.append(err)
    omega_history.append(omega.copy())
    torque_history.append(torque.copy())
    bias_history.append(est_bias.copy())
    rw_speed_history.append(rw_speed)
    disturbance_history.append(disturbance.copy())
  # Convert history lists to numpy arrays (AFTER LOOP)
omega_history = np.array(omega_history)
torque_history = np.array(torque_history)
bias_history = np.array(bias_history)   

print("Omega history shape:", omega_history.shape) 
print("torque history shape:",torque_history.shape)
print("bias history shape:",bias_history.shape)      
# ----------------------------------------
# PLOT
# ----------------------------------------

plt.figure(1)
plt.plot(time, error_history)
plt.axvline(40, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Attitude Error (deg)")
plt.title("Stable Full ADCS Simulation with Failure Mode")
plt.grid()
plt.show()
plt.figure(2)
plt.plot(time, omega_history[:,0])
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Angular Velocity vs Time")
plt.grid()
plt.show()
plt.figure(3)
plt.plot(time, torque_history[:,0])
plt.xlabel("Time (s)")
plt.ylabel("Control Torque (Nm)")
plt.title("Control Torque vs Time")
plt.grid()
plt.show()
plt.figure(4)
plt.plot(time, bias_history[:,0])
plt.xlabel("Time (s)")
plt.ylabel("Estimated Gyro Bias")
plt.title("Gyro Bias Estimation")
plt.grid()
plt.show()
print("Final Attitude Error (deg):", round(error_history[-1],5))  
