import numpy as np
from scipy.optimize import linear_sum_assignment

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

class Track(object):
    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount
        self.KF = ExtendedKalmanFilter()
        self.prediction = np.asarray(prediction)
        self.skipped_frames = 0
        self.trace = []
        self.confirmed = False  # Initialize confirmation status

class Tracker(object):
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 deletion_threshold, confirmation_threshold, trackIdCount,
                 min_assigned_detections, confirmation_frame_count):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.deletion_threshold = deletion_threshold
        self.confirmation_threshold = confirmation_threshold
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.min_assigned_detections = min_assigned_detections
        self.confirmation_frame_count = confirmation_frame_count

    def Update(self, detections):
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                diff = self.tracks[i].prediction - detections[j]
                distance = np.sqrt(diff[0][0]**2 + diff[1][0]**2)
                cost[i][j] = distance
        cost = (0.5) * cost

        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i in range(len(row_ind)):
            if cost[row_ind[i]][col_ind[i]] < self.dist_thresh:
                assignment[row_ind[i]] = col_ind[i]
            else:
                assignment[row_ind[i]] = -1

        un_assigned_tracks = [i for i, a in enumerate(assignment) if a == -1]

        for i in un_assigned_tracks:
            self.tracks[i].skipped_frames += 1

        del_tracks = [i for i, t in enumerate(self.tracks) if t.skipped_frames > self.max_frames_to_skip]

        for id in sorted(del_tracks, reverse=True):
            del self.tracks[id]
            del assignment[id]

        un_assigned_detects = [i for i in range(M) if i not in assignment]

        for i in un_assigned_detects:
            track = Track(detections[i], self.trackIdCount)
            self.trackIdCount += 1
            self.tracks.append(track)

        for i in range(len(assignment)):
            self.tracks[i].KF.predict()
            if assignment[i] != -1:
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]], 1)
                self.tracks[i].skipped_frames = 0
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(np.array([[0], [0]]), 0)

            self.tracks[i].trace.append(self.tracks[i].prediction)
            if len(self.tracks[i].trace) > self.max_trace_length:
                self.tracks[i].trace = self.tracks[i].trace[-self.max_trace_length:]

        for track in self.tracks:
            track_length = len(track.trace)
            if track_length >= self.confirmation_frame_count and not track.confirmed:
                assigned_count = sum(1 for i in assignment if i == track.track_id)
                if assigned_count >= self.min_assigned_detections:
                    track.confirmed = True

