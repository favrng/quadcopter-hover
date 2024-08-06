# quadcopter-hover

Program to control a quadcopter to hover. Made as a submission for bachelor thesis project.

Altitude control (z): Done using state feedback.
Attitude control (roll and pitch): Implemented using RL-based auto-tuning PD controller.

Cases:
1. Roll only (pitch torque is zero)
    Action space: Kp roll, Kd roll (dim = 2)
    Observation space: roll, roll dot, x angular velocity (dim = 3)
2. Pitch only (roll torque is zero)
    Action space: Kp pitch, Kd pitch (dim = 2)
    Observation space: pitch, pitch dot, y angular velocity (dim = 3)
3. Roll and pitch
    Action space: Kp roll, Kd roll, Kp pitch, Kd pitch (dim = 4)
    Observation space: roll, pitch, roll dot, pitch dot, x angular velocity, y angular velocity (dim = 6)

Trained models that are used for testing of case 1 and 2 are the same, due to the similarities of action and observation spaces.

