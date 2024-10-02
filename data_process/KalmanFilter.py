# import numpy as np
# from numpy import dot

# class KalmanFilterClass(object):
#     def __init__(self):
#         self.dt = 5e-3
#         self.A = np.array([[1,0],[0,1]])
#         self.u = np.zeros((2,1))
#         self.b = np.array([[0],[255]])
#         self.P = np.diag((3.0, 3.0))
#         self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
#         self.Q = np.eye(self.u.shape[0])
#         self.R = np.eye(self.b.shape[0])
#         self.lastResult = np.array([[0], [255]])
       
#     def predict(self):
#         self.u = np.round(dot(self.F, self.u))
#         self.P = dot(self.F, dot(self.P, self.F.T)) + self.Q
#         self.lastResult = self.u
#         return self.u

#     def correct(self, b, flag):
#         if not flag:
#             self.b = self.lastResult
#         else:
#             self.b = b
#         C = dot(self.A, dot(self.P, self.A.T))+self.R
#         K = dot(self.P, dot(self.A.T, np.linalg.inv(C)))
#         self.u = np.round(self.u + dot(K, (self.b - dot(self.A, self.u))))
#         self.P = self.P - dot(K, dot(C, K.T))
#         self.lastResult = self.u
#         return self.u
       
import numpy as np

class ExtendedKalmanFilter(object):
    def __init__(self):
        self.dt = 5e-3  # Time step

        # Define state transition matrix A assuming it might be replaced
        # by a nonlinear function for prediction step.
        self.A = np.array([[1, 0], [0, 1]])

        # Initial state
        self.x = np.zeros((2, 1))

        # External control input (if any)
        self.u = np.zeros((2, 1))

        # Process noise covariance
        self.Q = np.eye(self.x.shape[0])

        # Measurement noise covariance
        self.R = np.eye(2)  # Adjust dimension as per measurement vector

        # Measurement matrix, assuming it might be replaced by a nonlinear
        # function for update step.
        self.H = np.array([[1, 0], [0, 1]])

        # Covariance matrix
        self.P = np.diag((3.0, 3.0))

        # Last estimated state
        self.last_x = np.array([[0], [255]])

    def predict(self):
        # State transition function f(x) should be defined here,
        # for linear case it is equivalent to dot(A, x)
        # Assuming f(x) = Ax for simplicity; replace with actual function if needed.
        self.x = np.dot(self.A, self.x) + self.u

        # Compute Jacobian of F at the last estimated state for linear approximation
        F_j = np.array([[1.0, self.dt], [0.0, 1.0]])  # Placeholder for actual Jacobian

        # Predict process covariance
        self.P = np.dot(F_j, np.dot(self.P, F_j.T)) + self.Q

        # Store current state as last estimated state for future iterations
        self.last_x = self.x.copy()
        return self.x

    def correct(self, z, flag):
        # Measurement update
        # Measurement function h(x) should be defined here.
        # For linear case it's equivalent to dot(H, x).
        # Assuming h(x) = Hx for simplicity; replace with actual function if needed.
        if not flag:
            z = np.dot(self.H, self.last_x)

        # Compute Jacobian of H at the last estimated state
        H_j = self.H  # Placeholder for actual Jacobian

        # Calculate Kalman gain
        S = np.dot(H_j, np.dot(self.P, H_j.T)) + self.R
        K = np.dot(self.P, np.dot(H_j.T, np.linalg.inv(S)))

        # Update state estimate
        y = z - np.dot(H_j, self.x)
        self.x = self.x + np.dot(K, y)

        # Update covariance estimate
        self.P = self.P - np.dot(K, np.dot(H_j, self.P))

        # Store current estimate as last estimate
        self.last_x = self.x.copy()
        return self.x