import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# ---------------------------
# System Dynamics (Inverted Pendulum on Cart)
# ---------------------------
class InvertedPendulum:
    def __init__(self, M=1.0, m=0.1, b=0.1, I=0.006, g=9.8, l=0.3):
        self.M = M  # Mass of cart
        self.m = m  # Mass of pendulum
        self.b = b  # friction
        self.I = I  # inertia
        self.g = g  # gravity
        self.l = l  # length to pendulum center of mass

    def dynamics(self, state, u):
        # state = [x, x_dot, theta, theta_dot]
        x, x_dot, theta, theta_dot = state
        M, m, b, I, g, l = self.M, self.m, self.b, self.I, self.g, self.l

        Sy = np.sin(theta)
        Cy = np.cos(theta)
        D = I*(M+m) + M*m*l**2

        x_ddot = (1/D)*(
            -m**2*l**2*g*Cy*Sy +
            (I+m*l**2)*(u - b*x_dot + m*l*theta_dot**2*Sy)
        )
        theta_ddot = (1/D)*(
            (m+M)*m*g*l*Sy -
            (u - b*x_dot + m*l*theta_dot**2*Sy)*m*l*Cy
        )
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


# ---------------------------
# RK4 Integrator
# ---------------------------
def rk4_step(dynamics, state, u, dt):
    k1 = dynamics(state, u)
    k2 = dynamics(state + 0.5*dt*k1, u)
    k3 = dynamics(state + 0.5*dt*k2, u)
    k4 = dynamics(state + dt*k3, u)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ---------------------------
# PID Controller
# ---------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, dt=0.02):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error):
        self.integral += error*self.dt
        derivative = (error - self.prev_error)/self.dt
        self.prev_error = error
        return self.Kp*error + self.Ki*self.integral + self.Kd*derivative


# ---------------------------
# Simulation
# ---------------------------
def simulate(pendulum, controller, t_max=10, dt=0.02):
    state = np.array([0.0, 0.0, np.pi/6, 0.0])  # start tilted 30 deg
    t_vals, theta_vals = [], []
    perf = 0.0
    for i in range(int(t_max/dt)):
        t = i*dt
        error = -state[2]  # want theta=0
        u = controller.control(error)
        state = rk4_step(pendulum.dynamics, state, u, dt)
        t_vals.append(t)
        theta_vals.append(state[2])
        perf += abs(error)*dt
    return np.array(t_vals), np.array(theta_vals), perf


# ---------------------------
# Generate Dataset
# ---------------------------
def generate_dataset(pendulum, n_samples=200):
    X, y = [], []
    for _ in range(n_samples):
        Kp = np.random.uniform(10, 200)
        Ki = np.random.uniform(0, 20)
        Kd = np.random.uniform(0, 50)
        pid = PID(Kp, Ki, Kd)
        _, _, perf = simulate(pendulum, pid, t_max=5)
        X.append([Kp, Ki, Kd])
        y.append(perf)
    return np.array(X), np.array(y)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    pendulum = InvertedPendulum()

    # Hand-tuned PID
    pid_hand = PID(Kp=80, Ki=1, Kd=20)
    t1, theta1, perf_hand = simulate(pendulum, pid_hand)

    # Dataset for ML
    X, y = generate_dataset(pendulum, 300)
    np.savetxt("pid_dataset.csv", np.hstack([X, y.reshape(-1,1)]),
               delimiter=",", header="Kp,Ki,Kd,Perf", comments="")

    # Train ML model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLPRegressor(hidden_layer_sizes=(32,32), max_iter=2000).fit(X_train, y_train)
    print("Train MSE:", np.mean((model.predict(X_train)-y_train)**2))
    print("Test MSE:", np.mean((model.predict(X_test)-y_test)**2))

    # Use ML to suggest gains (search space)
    test_candidates = np.array([[np.random.uniform(10,200),
                                 np.random.uniform(0,20),
                                 np.random.uniform(0,50)] for _ in range(200)])
    preds = model.predict(test_candidates)
    best_idx = np.argmin(preds)
    best_gains = test_candidates[best_idx]
    pid_ml = PID(*best_gains)
    t2, theta2, perf_ml = simulate(pendulum, pid_ml)

    # Plot
    plt.figure()
    plt.plot(t1, theta1, label="Hand-tuned PID")
    plt.plot(t2, theta2, label="ML-suggested PID")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.legend()
    plt.grid()
    plt.savefig("inverted_pendulum_comparison.png")

    # Report
    report_text = f"""
Inverted pendulum on a cart - simulation report
-----------------------------------------------

Hand-tuned PID:
  Kp={pid_hand.Kp}, Ki={pid_hand.Ki}, Kd={pid_hand.Kd}
  -> performance metric = {perf_hand:.3f}

ML-tuned PID:
  Kp={pid_ml.Kp:.2f}, Ki={pid_ml.Ki:.2f}, Kd={pid_ml.Kd:.2f}
  -> performance metric = {perf_ml:.3f}

Comparison plot saved as inverted_pendulum_comparison.png
Dataset saved as pid_dataset.csv
"""

    with open("simulation_report.txt", "w") as f:
        f.write(report_text)

    print("Report written to simulation_report.txt")
