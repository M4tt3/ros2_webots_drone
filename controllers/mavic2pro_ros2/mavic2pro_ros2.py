"""Ideal (kinematic) drone + fully emulated Intel RealSense D435i, on ROS 2.

Command interface
    /drone/cmd_vel   (geometry_msgs/Twist)        linear.x/y/z + angular.z, body (FLU) frame
                     velocity setpoint. If no cmd_vel message arrives for cmd_vel_timeout seconds, 
                     velocity decays to zero (hover) rather than continuing to fly the last command.

    ros2 topic pub -r 20 /drone/cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.2}}"

Published topics
    /drone/odom                  nav_msgs/Odometry      ground-truth pose + twist
    /d435i/color/image_raw       sensor_msgs/Image      rgb8
    /d435i/color/camera_info     sensor_msgs/CameraInfo
    /d435i/infra1/image_raw      sensor_msgs/Image      mono8 (left IR + noise, dots if projector_enabled)
    /d435i/infra1/camera_info    sensor_msgs/CameraInfo
    /d435i/infra2/image_raw      sensor_msgs/Image      mono8 (right IR + noise, dots if projector_enabled)
    /d435i/infra2/camera_info    sensor_msgs/CameraInfo
    /d435i/depth/image_rect_raw  sensor_msgs/Image      32FC1 [m], SGBM on the IR pair
    /d435i/depth/camera_info     sensor_msgs/CameraInfo
    /d435i/depth_gt/image_raw    sensor_msgs/Image      32FC1 [m], RangeFinder ground truth
    /drone/imu                   sensor_msgs/Imu        synthesized BMI055 (accel + gyro)
    /tf, /clock
"""

import array as pyarray
import math
import sys

import cv2
import numpy as np

from controller import Supervisor

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage
from transforms3d import quaternions, euler

from std_msgs.msg import Bool

import threading

PRINT_STATUS = True

# Camera parameters
IMAGE_RATE = 15.0
STEREO_BASELINE = 0.05          # 50 mm stereo baseline
PROJECTOR_POWER = 0.0  # dot brightness at 1 m (falls off as 1/Z^2)
PROJECTOR_MAX_RANGE = 8.0
GRAVITY = 9.81
ENABLE_STEREO_DEPTH = True

# Noise
IMU_ACCEL_NOISE = 0.02
IMU_GYRO_NOISE = 0.004
IR_NOISE_SIGMA = 3.0  # mono8 pixel noise std, for emulated IR images

# Vel command parameters for attitude calculation
MAX_TILT_RAD = math.radians(80)  # max roll/pitch disturbance to command
TILT_GAIN = 2.0
VERTICAL_STEP = 0.01

# Attitude stabilization (quad-X motor mixing), same structure as Webots'
# stock Mavic 2 Pro controller: a self-leveling P term on the measured
# roll/pitch, damped by the measured angular rate, biased by the commanded
# disturbance from cmd_vel_to_attitude.
ROLL_P = 50.0
PITCH_P = 30.0
VERTICAL_P = 3.0
THRUST_BASELINE = 60
K_VERTICAL_OFFSET = 0.6  # add to altitude error to avoid oscillating around zero
ATTITUDE_CLAMP_RAD = 1.0  # saturate the measured roll/pitch fed into the P terms
TAKEOFF_K = 3.0

ROLL_BIAS = 0.0
PITCH_BIAS = 0.0
YAW_BIAS = 0.0

# Ground truth odom
# Transform from Webots world frame (X forward, Y left, Z up) to ROS odom
# frame (X backward, Y right, Z up): a 180 degree rotation about Z, so x and
# y both flip sign and z is unchanged.
T_WORLD_TO_ODOM = quaternions.axangle2quat([0.0, 0.0, 1.0], math.pi)  # (w, x, y, z)


def world_to_odom_vec(x, y, z):
    """Rotate a world-frame vector (position, linear/angular velocity) into the odom frame."""
    return -x, -y, z


def world_to_odom_quat(qw, qx, qy, qz):
    """Rotate a world-frame orientation quaternion into the odom frame."""
    ow, ox, oy, oz = quaternions.qmult(T_WORLD_TO_ODOM, (qw, qx, qy, qz))
    return ox, oy, oz, ow


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

class DroneControllerNode(Node):
    def __init__(self, supervisor: Supervisor):
        super().__init__("webots_mavic2pro_controller")

        dt = supervisor.getBasicTimeStep()

        self.supervisor = supervisor
        self.motors = {
            "front_left": self.supervisor.getDevice("front left propeller"),
            "front_right": self.supervisor.getDevice("front right propeller"),
            "rear_left": self.supervisor.getDevice("rear left propeller"),
            "rear_right": self.supervisor.getDevice("rear right propeller")
        }

        # Set to velocity mode
        self.motors["front_left"].setPosition(math.inf)
        self.motors["front_right"].setPosition(math.inf)
        self.motors["rear_left"].setPosition(math.inf)
        self.motors["rear_right"].setPosition(math.inf)

        self.imu = self.supervisor.getDevice("inertial unit")
        self.gyro = self.supervisor.getDevice("gyro")
        self.accel = self.supervisor.getDevice("accelerometer")
        self.gps = self.supervisor.getDevice("gps")
        self.imu.enable(int(dt))
        self.gyro.enable(int(dt))
        self.accel.enable(int(dt))
        self.gps.enable(int(dt))

        self.cmd_vel = np.zeros(4)  # x, y, z, yaw_rate in body frame
        self.last_cmd_vel_time = -math.inf
        self.target_altitude = 1.5
        self.sim_time = 0.0  # kept in sync by the main loop, used to time out stale cmd_vel

        self.create_subscription(Twist, "/drone/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Bool, "/drone/arm", self.on_arm, 10)

        self.pub_odom = self.create_publisher(Odometry, "/drone/odom", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_imu = self.create_publisher(Imu, "/drone/imu", qos_profile_sensor_data)
        
        #self.timer = self.create_timer(dt / 1000.0, self.control_loop)  # 100 Hz control loop

        self.takeoff = False
        self.armed = False

        print("Mavic 2 Pro ROS2 controller ready, waiting for /drone/cmd_vel and /drone/arm messages...")

    def apply_noise_imu(self, accel, gyro):
        return (accel + np.random.normal(0.0, IMU_ACCEL_NOISE, 3),
                gyro + np.random.normal(0.0, IMU_GYRO_NOISE, 3))
    
    def publish_imu(self):
        accel = np.array(self.accel.getValues())
        gyro = np.array(self.gyro.getValues())
        accel, gyro = self.apply_noise_imu(accel, gyro)

        imu_msg = Imu()
        imu_msg.header.stamp = to_stamp(self.sim_time)
        imu_msg.header.frame_id = "base_link"
        imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = accel
        imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = gyro
        self.pub_imu.publish(imu_msg)

    def publish_odom(self):

        odom_msg = Odometry()
        odom_msg.header.stamp = to_stamp(self.sim_time)
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Get ground truth pose and twist from Webots
        x, y, z = self.supervisor.getSelf().getPosition()
        vx, vy, vz, wx, wy, wz = self.supervisor.getSelf().getVelocity()
        R = np.matrix(self.supervisor.getSelf().getOrientation())
        R.reshape((3, 3))
        qw, qx, qy, qz = quaternions.mat2quat(R)

        # Transform ground truth from the Webots world frame into the odom frame
        x, y, z = world_to_odom_vec(x, y, z)
        vx, vy, vz = world_to_odom_vec(vx, vy, vz)
        wx, wy, wz = world_to_odom_vec(wx, wy, wz)
        qx, qy, qz, qw = world_to_odom_quat(qw, qx, qy, qz)

        # Position and velocity
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.twist.twist.linear.x = vx
        odom_msg.twist.twist.linear.y = vy
        odom_msg.twist.twist.linear.z = vz

        # Orientation and angular velocity
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.angular.x = wx
        odom_msg.twist.twist.angular.y = wy
        odom_msg.twist.twist.angular.z = wz

        self.pub_odom.publish(odom_msg)

    def get_current_attitude(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        return roll, pitch, yaw

    def get_current_rates(self):
        """Body-frame roll/pitch/yaw rates, approximated by the world-frame
        angular velocity (valid near-level, consistent with the axis-angle
        roll/pitch/yaw approximation in get_ground_truth_pose)."""
        wx, wy, wz = self.gyro.getValues()
        return wx, wy, wz
    
    def get_current_pos(self):
        return self.gps.getValues()
    
    def get_current_vel(self):
        return self.gps.getSpeedVector()

    def cmd_vel_to_attitude(self, cmd_vel):
        """
        Convert a body-frame velocity command into roll/pitch/yaw attitude
        disturbances, using the standard small-angle approximation for
        quadrotor velocity control.

        cmd_vel: (vx, vy, vz, yaw_rate)
            vx, vy  -> body-frame forward/lateral velocity command [m/s]
            vz      -> vertical velocity command [m/s] (handled by thrust,
                    not rpy — passed through untouched here)
            yaw_rate -> commanded yaw rate [rad/s]

        Returns: (roll_attitude, pitch_attitude, yaw_attitude, thrust)
            roll/pitch/yaw attitudes are in radians, meant to be added to
            (or tracked as a setpoint by) the attitude controller.
        """
        vx, vy, _, yaw_rate = cmd_vel

        # Forward velocity -> pitch (nose-down to accelerate forward)
        # Lateral velocity -> roll (bank to accelerate sideways)
        # Sign convention: positive vx -> negative pitch (nose down), NED-style.
        # Flip signs here if your sim uses a different convention.
        pitch_dist = -TILT_GAIN * vx
        roll_dist  =  TILT_GAIN * vy

        # Clamp to max tilt to keep the small-angle approx valid and avoid
        # commanding unrealistic attitudes
        pitch_dist = np.clip(pitch_dist, -MAX_TILT_RAD, MAX_TILT_RAD)
        roll_dist  = np.clip(roll_dist,  -MAX_TILT_RAD, MAX_TILT_RAD)

        # Yaw is direct — no coupling through attitude needed, it's already
        # a rotational rate command
        yaw_dist = yaw_rate

        return roll_dist, pitch_dist, yaw_dist

    def on_cmd_vel(self, msg: Twist):
        self.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])
        self.last_cmd_vel_time = self.sim_time

    def on_arm(self, msg: Bool):
        self.armed = msg.data
        if not self.armed:
            # Disarm: stop motors and reset target altitude
            for motor in self.motors.values():
                motor.setVelocity(0.0)
            self.takeoff = False

    def control_loop(self):

        if not self.armed:
            for motor in self.motors.values():
                motor.setVelocity(0.0)
            return

        roll_dist, pitch_dist, yaw_dist = self.cmd_vel_to_attitude(self.cmd_vel)

        self.target_altitude += self.cmd_vel[2] * VERTICAL_STEP  # integrate vertical velocity command into a target altitude
        altitude = self.get_current_pos()[2]

        roll, pitch, _ = self.get_current_attitude()
        roll_rate, pitch_rate, yaw_rate = self.get_current_rates()

        roll_input = (ROLL_P * np.clip(roll, -ATTITUDE_CLAMP_RAD, ATTITUDE_CLAMP_RAD)
                        + roll_rate + roll_dist + ROLL_BIAS)
        pitch_input = (PITCH_P * np.clip(pitch, -ATTITUDE_CLAMP_RAD, ATTITUDE_CLAMP_RAD)
                        + pitch_rate + pitch_dist + PITCH_BIAS)
        yaw_input = yaw_dist + YAW_BIAS

        clamped_difference_altitude = np.clip(self.target_altitude - altitude + K_VERTICAL_OFFSET, -1.0, 1.0);
        vertical_input = clamped_difference_altitude ** 3 * VERTICAL_P  # simple P controller for altitude

        if not self.takeoff:
            self.takeoff = altitude > self.target_altitude * 0.5
            vertical_input = vertical_input * TAKEOFF_K  # boost thrust during takeoff to get off the ground

        # Standard quad-X mixing (front left/rear right vs. front right/rear
        # left spin in opposite directions, hence the sign flips below).
        front_left = THRUST_BASELINE + vertical_input - roll_input + pitch_input - yaw_input;
        front_right = THRUST_BASELINE + vertical_input + roll_input + pitch_input + yaw_input;
        rear_left = THRUST_BASELINE + vertical_input - roll_input - pitch_input + yaw_input;
        rear_right = THRUST_BASELINE + vertical_input + roll_input - pitch_input - yaw_input;

        if PRINT_STATUS:
            print(f"STATUS:             roll: {roll:.3f}, pitch: {pitch:.3f}, yaw: {yaw_rate:.3f} \n"
                  f"VELOCITY_CMD:       vx: {self.cmd_vel[0]:.3f}, vy: {self.cmd_vel[1]:.3f}, vz: {self.cmd_vel[2]:.3f}, yaw_rate: {self.cmd_vel[3]:.3f} \n"
                  f"TARGET_ATTITUDE:    roll_dist: {roll_dist:.3f}, pitch_dist: {pitch_dist:.3f}, yaw_dist: {yaw_dist:.3f} \n"
                  f"ATTITUDE_CMDS:      roll_input: {roll_input:.3f}, pitch_input: {pitch_input:.3f}, yaw_input: {yaw_input:.3f}, vertical_input: {vertical_input:.3f} \n"
                  f"ALTITUDE:           target_altitude: {self.target_altitude:.3f}, takeoff: {self.takeoff}\n"
                  f"MOTORS:             front_right: {front_right:.3f}, front_left: {front_left:.3f}, rear_right: {rear_right:.3f}, rear_left: {rear_left:.3f}\n---------------------------\n")

        self.motors["front_right"].setVelocity(-front_right)
        self.motors["front_left"].setVelocity(front_left)
        self.motors["rear_right"].setVelocity(rear_right)
        self.motors["rear_left"].setVelocity(-rear_left)


class CameraPublisherNode(Node):
    def __init__(self, supervisor: Supervisor):
        super().__init__("camera_publisher_node")
        self.pub_color = self.create_publisher(Image, "/d435i/color/image_raw", 10)
        self.pub_color_info = self.create_publisher(CameraInfo, "/d435i/color/camera_info", 10)
        self.pub_infra1 = self.create_publisher(Image, "/d435i/infra1/image_raw", 10)
        self.pub_infra1_info = self.create_publisher(CameraInfo, "/d435i/infra1/camera_info", 10)
        self.pub_infra2 = self.create_publisher(Image, "/d435i/infra2/image_raw", 10)
        self.pub_infra2_info = self.create_publisher(CameraInfo, "/d435i/infra2/camera_info", 10)
        self.pub_depth = self.create_publisher(Image, "/d435i/depth/image_rect_raw", 10)
        self.pub_depth_info = self.create_publisher(CameraInfo, "/d435i/depth/camera_info", 10)
        self.pub_depth_gt = self.create_publisher(Image, "/d435i/depth_gt/image_raw", 10)

        self.supervisor = supervisor
        self.vision_period_ms = max(supervisor.getBasicTimeStep(), int(round(1000.0 / IMAGE_RATE)))

        self.rgb_cam = supervisor.getDevice("d435i_rgb")
        self.ir_left_cam = supervisor.getDevice("d435i_ir_left")
        self.ir_right_cam = supervisor.getDevice("d435i_ir_right")
        self.depth_gt_sensor = supervisor.getDevice("d435i_depth")

        self.rgb_cam.enable(self.vision_period_ms)
        self.ir_left_cam.enable(self.vision_period_ms)
        self.ir_right_cam.enable(self.vision_period_ms)
        self.depth_gt_sensor.enable(self.vision_period_ms)

        self.ir_width = self.ir_left_cam.getWidth()
        self.ir_height = self.ir_left_cam.getHeight()
        self.rgb_width = self.rgb_cam.getWidth()
        self.rgb_height = self.rgb_cam.getHeight()
        self.depth_width = self.depth_gt_sensor.getWidth()
        self.depth_height = self.depth_gt_sensor.getHeight()

        self.rng = np.random.default_rng(435)
        n_dots = int(0.04 * self.ir_width * self.ir_height)
        self.emitter_dot_x = self.rng.integers(1, self.ir_width - 1, n_dots)
        self.emitter_dot_y = self.rng.integers(1, self.ir_height - 1, n_dots)

        self.fx_ir = (self.ir_width / 2.0) / math.tan(self.ir_left_cam.getFov() / 2.0)
        self.fx_rgb = (self.rgb_width / 2.0) / math.tan(self.rgb_cam.getFov() / 2.0)
        self.max_range = self.depth_gt_sensor.getMaxRange()

        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # min stereo depth = FX_IR * B / 128 (~0.17 m at 848 px)
            blockSize=5,
            P1=8 * 5 * 5,
            P2=32 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=120,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        print(f"D435i/ROS2 ready: RGB {self.rgb_width}x{self.rgb_height} | IR/depth {self.ir_width}x{self.ir_height}, "
          f"fx={self.fx_ir:.1f}px, baseline={STEREO_BASELINE * 1000:.0f}mm")

    def read_depth_gt(self):
        # data_type='buffer' returns a raw ctypes float pointer: wrap it
        # zero-copy, then copy out of Webots-owned memory.
        raw = self.depth_gt_sensor.getRangeImage(data_type="buffer")
        depth = np.ctypeslib.as_array(raw, (self.ir_height * self.ir_width,)).reshape((self.ir_height, self.ir_width)).copy()
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def emulate_ir_pair(self, depth_gt, noise_sigma):
        left = cv2.cvtColor(bgra(self.ir_left_cam, self.ir_width, self.ir_height), cv2.COLOR_BGRA2GRAY).astype(np.float32)
        right = cv2.cvtColor(bgra(self.ir_right_cam, self.ir_width, self.ir_height), cv2.COLOR_BGRA2GRAY).astype(np.float32)

        dot_x, dot_y = self.emitter_dot_x, self.emitter_dot_y
        z = depth_gt[dot_y, dot_x]
        lit = (z > 0.1) & (z < self.max_range)
        intensity = np.clip(PROJECTOR_POWER / (z * z + 1e-6), 0, 180)

        # left image: dots land where the (left-aligned) depth map says they are
        np.add.at(left, (dot_y[lit], dot_x[lit]), intensity[lit])
        # right image: same dots shifted by the true disparity fx * B / Z
        disparity = self.fx_ir * STEREO_BASELINE / np.maximum(z, 1e-6)
        xr = np.rint(dot_x - disparity).astype(np.int32)
        ok = lit & (xr >= 0)
        np.add.at(right, (dot_y[ok], xr[ok]), intensity[ok])

        left += self.rng.standard_normal(left.shape, dtype=np.float32) * noise_sigma
        right += self.rng.standard_normal(right.shape, dtype=np.float32) * noise_sigma
        return (np.clip(left, 0, 255).astype(np.uint8),
                np.clip(right, 0, 255).astype(np.uint8))
    
    def stereo_depth(self, left_ir, right_ir):
        disparity = self.stereo_matcher.compute(left_ir, right_ir).astype(np.float32) / 16.0
        depth = np.zeros_like(disparity)
        valid = disparity > 0.5
        depth[valid] = self.fx_ir * STEREO_BASELINE / disparity[valid]
        depth[depth > self.max_range] = 0.0
        return depth
    

    def publish_images(self):
        depth_gt = self.read_depth_gt()
        left_ir, right_ir = self.emulate_ir_pair(depth_gt, IR_NOISE_SIGMA)
        rgb = cv2.cvtColor(bgra(self.rgb_cam, self.rgb_width, self.rgb_height), cv2.COLOR_BGRA2RGB)
        stamp = to_stamp(self.supervisor.getTime())

        self.pub_color.publish(make_image_msg(
            stamp, "d435i_color_optical_frame", rgb, "rgb8", 3))
        self.pub_color_info.publish(make_camera_info(
            stamp, "d435i_color_optical_frame", self.rgb_width, self.rgb_height, self.fx_rgb))
        self.pub_infra1.publish(make_image_msg(
            stamp, "d435i_infra1_optical_frame", left_ir, "mono8", 1))
        self.pub_infra1_info.publish(make_camera_info(
            stamp, "d435i_infra1_optical_frame", self.ir_width, self.ir_height, self.fx_ir))
        self.pub_infra2.publish(make_image_msg(
            stamp, "d435i_infra2_optical_frame", right_ir, "mono8", 1))
        self.pub_infra2_info.publish(make_camera_info(
            stamp, "d435i_infra2_optical_frame", self.ir_width, self.ir_height, self.fx_ir,
            tx=-self.fx_ir * STEREO_BASELINE))
        self.pub_depth_gt.publish(make_image_msg(
            stamp, "d435i_depth_optical_frame", depth_gt, "32FC1", 4))

        if ENABLE_STEREO_DEPTH:
            depth = self.stereo_depth(left_ir, right_ir)
            self.pub_depth.publish(make_image_msg(
                stamp, "d435i_depth_optical_frame", depth, "32FC1", 4))
            self.pub_depth_info.publish(make_camera_info(
                stamp, "d435i_depth_optical_frame", self.ir_width, self.ir_height, self.fx_ir))

def bgra(camera, width, height):
    return np.frombuffer(camera.getImage(), np.uint8).reshape((height, width, 4))

def to_stamp(sim_time):
    sec = int(sim_time)
    return Time(sec=sec, nanosec=int(round((sim_time - sec) * 1e9)))

def make_image_msg(stamp, frame_id, array, encoding, bytes_per_pixel):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * bytes_per_pixel
    # array.array takes rclpy's fast setter branch; raw bytes would trigger a
    # per-element __debug__ validation loop over megabytes of data.
    msg.data = pyarray.array("B", array.tobytes())
    return msg


def make_camera_info(stamp, frame_id, width, height, fx, tx=0.0):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    cx, cy = width / 2.0, height / 2.0
    msg.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, tx, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main():
    supervisor = Supervisor()

    rclpy.init(args=sys.argv)
    controller_node = DroneControllerNode(supervisor)
    realsense_node = CameraPublisherNode(supervisor)

    spin_drone = lambda: rclpy.spin(controller_node)

    drone_controller_thread = threading.Thread(target=spin_drone, daemon=True)
    drone_controller_thread.start()

    last_camera_publish_time = -np.inf

    while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
        controller_node.sim_time = supervisor.getTime()
        controller_node.pub_clock.publish(Clock(clock=to_stamp(controller_node.sim_time)))
        controller_node.control_loop()
        controller_node.publish_imu()
        controller_node.publish_odom()
        if supervisor.getTime() - last_camera_publish_time >= realsense_node.vision_period_ms / 1000.0:
            realsense_node.publish_images()
            last_camera_publish_time = supervisor.getTime()
    
    controller_node.destroy_node()
    realsense_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
