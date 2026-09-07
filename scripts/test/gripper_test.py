import rtde_control
import rtde_io
import time

ROBOT_IP = "192.168.2.155"   # Change to your robot IP

rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_io_ = rtde_io.RTDEIOInterface(ROBOT_IP)

try:
    print("Connected.")

    print("DO0 ON")
    rtde_io_.setToolDigitalOut(1, True)

finally:
    rtde_c.stopScript()
