"""Ideal (kinematic) drone + fully emulated Intel RealSense D435i, on ROS 2.

The robot is a Supervisor without physics: every control step the controller
moves the drone toward the commanded x/y/z/yaw pose with velocity-limited,
optionally noise-injected motion, by writing the translation/rotation fields
directly. No propellers, no dynamics - commands are executed "ideally".

Command interface
    /drone/cmd_pose  (geometry_msgs/PoseStamped)  x, y, z + yaw (from quaternion)
                     absolute position setpoint; switches the drone to position-hold mode.
    /drone/cmd_vel   (geometry_msgs/Twist)        linear.x/y/z + angular.z, body (FLU) frame
                     velocity setpoint; switches the drone to velocity mode. If no cmd_vel
                     message arrives for cmd_vel_timeout seconds, velocity decays to zero (hover)
                     rather than continuing to fly the last command.

    ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
        "{pose: {position: {x: 3.0, y: 1.0, z: 2.0}, orientation: {w: 1.0}}}"

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
    /d435i/imu                   sensor_msgs/Imu        synthesized BMI055 (accel + gyro)
    /tf, /clock

Parameters (override via controllerArgs in the world file, e.g.
["--ros-args", "-p", "max_speed_xy:=5.0"], or at runtime with `ros2 param set`)
    max_speed_xy      [m/s]   horizontal speed limit          (default 2.0)
    max_speed_z       [m/s]   vertical speed limit            (default 1.5)
    max_yaw_rate      [rad/s] yaw rate limit                  (default 1.5)
    pos_gain          [1/s]   P gain position -> velocity     (default 1.5)
    yaw_gain          [1/s]   P gain yaw -> yaw rate          (default 2.0)
    cmd_vel_timeout   [s]     hover if cmd_vel goes stale for this long (default 0.5)
    motion_noise_xy   [m/s]   std of horizontal velocity noise (default 0.03)
    motion_noise_z    [m/s]   std of vertical velocity noise   (default 0.02)
    motion_noise_yaw  [rad/s] std of yaw rate noise            (default 0.01)
    imu_accel_noise   [m/s^2] accelerometer noise std          (default 0.05)
    imu_gyro_noise    [rad/s] gyroscope noise std              (default 0.005)
    ir_noise_sigma    [grey]  IR image read noise std          (default 4.0)
    projector_enabled [bool]  project the IR dot pattern       (default False)
    image_rate        [Hz]    camera/depth publish rate        (default 15.0)
    enable_stereo_depth [bool] run SGBM and publish /d435i/depth (default true)
    camera_xyz        [m,m,m] D435i mount offset on base_link - must match the
                              sensorSlot translation in the world (default [0.06, 0, -0.005])
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

BASELINE = 0.05          # 50 mm stereo baseline
PROJECTOR_POWER = 500.0  # dot brightness at 1 m (falls off as 1/Z^2)
PROJECTOR_MAX_RANGE = 8.0
GRAVITY = 9.81
# FLU camera body frame -> ROS optical frame (z forward, x right, y down)
OPTICAL_QUAT = (-0.5, 0.5, -0.5, 0.5)  # x, y, z, w


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class IdealDroneNode(Node):
    def __init__(self):
        super().__init__("d435i_drone")
        p = self.declare_parameter
        p("max_speed_xy", 2.0)
        p("max_speed_z", 1.5)
        p("max_yaw_rate", 1.5)
        p("pos_gain", 1.5)
        p("yaw_gain", 2.0)
        p("cmd_vel_timeout", 0.5)
        p("motion_noise_xy", 0.03)
        p("motion_noise_z", 0.02)
        p("motion_noise_yaw", 0.01)
        p("imu_accel_noise", 0.05)
        p("imu_gyro_noise", 0.005)
        p("ir_noise_sigma", 4.0)
        p("projector_enabled", False)
        p("image_rate", 15.0)
        p("enable_stereo_depth", True)
        p("camera_xyz", [0.06, 0.0, -0.005])

        self.target_pos = None  # set from the initial pose on first step
        self.target_yaw = None
        self.control_mode = "position"  # or "velocity", switched by whichever topic last published
        self.cmd_vel_linear = np.zeros(3)   # body-frame [m/s]
        self.cmd_vel_yaw_rate = 0.0          # [rad/s]
        self.last_cmd_vel_time = -math.inf
        self.sim_time = 0.0  # kept in sync by the main loop, used to time out stale cmd_vel
        self.create_subscription(PoseStamped, "/drone/cmd_pose", self._on_cmd_pose, 10)
        self.create_subscription(Twist, "/drone/cmd_vel", self._on_cmd_vel, 10)

        self.pub_odom = self.create_publisher(Odometry, "/drone/odom", 10)
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_imu = self.create_publisher(Imu, "/d435i/imu", qos_profile_sensor_data)
        img_qos = qos_profile_sensor_data
        # ov_msckf's stereo message_filters::Subscriber uses the default (RELIABLE) QoS,
        # so infra1/infra2 must be published RELIABLE or the VIO subscriber never matches.
        stereo_img_qos = 10
        self.pub_color = self.create_publisher(Image, "/d435i/color/image_raw", img_qos)
        self.pub_color_info = self.create_publisher(CameraInfo, "/d435i/color/camera_info", qos_profile_sensor_data)
        self.pub_infra1 = self.create_publisher(Image, "/d435i/infra1/image_raw", stereo_img_qos)
        self.pub_infra1_info = self.create_publisher(CameraInfo, "/d435i/infra1/camera_info", qos_profile_sensor_data)
        self.pub_infra2 = self.create_publisher(Image, "/d435i/infra2/image_raw", stereo_img_qos)
        self.pub_infra2_info = self.create_publisher(CameraInfo, "/d435i/infra2/camera_info", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, "/d435i/depth/image_rect_raw", img_qos)
        self.pub_depth_info = self.create_publisher(CameraInfo, "/d435i/depth/camera_info", qos_profile_sensor_data)
        self.pub_depth_gt = self.create_publisher(Image, "/d435i/depth_gt/image_raw", img_qos)

    def _on_cmd_pose(self, msg: PoseStamped):
        pos = msg.pose.position
        self.target_pos = np.array([pos.x, pos.y, pos.z])
        self.target_yaw = yaw_from_quaternion(msg.pose.orientation)
        if self.control_mode != "position":
            self.get_logger().info("switching to position-hold mode")
        self.control_mode = "position"
        self.get_logger().info(
            f"new setpoint: x={pos.x:.2f} y={pos.y:.2f} z={pos.z:.2f} "
            f"yaw={math.degrees(self.target_yaw):.0f} deg")

    def _on_cmd_vel(self, msg: Twist):
        self.cmd_vel_linear = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        self.cmd_vel_yaw_rate = msg.angular.z
        self.last_cmd_vel_time = self.sim_time
        if self.control_mode != "velocity":
            self.get_logger().info("switching to velocity mode")
        self.control_mode = "velocity"


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


def make_transform(stamp, parent, child, xyz, quat):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    (t.transform.rotation.x, t.transform.rotation.y,
     t.transform.rotation.z, t.transform.rotation.w) = quat
    return t


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    dt = timestep / 1000.0

    rclpy.init(args=sys.argv)
    node = IdealDroneNode()
    param = lambda name: node.get_parameter(name).value

    # ---------------------------------------------------------- devices
    vision_period_ms = max(timestep, int(round(1000.0 / param("image_rate"))))
    rgb_cam = supervisor.getDevice("d435i_rgb")
    ir_left_cam = supervisor.getDevice("d435i_ir_left")
    ir_right_cam = supervisor.getDevice("d435i_ir_right")
    depth_gt_sensor = supervisor.getDevice("d435i_depth")
    for device in (rgb_cam, ir_left_cam, ir_right_cam, depth_gt_sensor):
        device.enable(vision_period_ms)

    IR_W, IR_H = ir_left_cam.getWidth(), ir_left_cam.getHeight()
    RGB_W, RGB_H = rgb_cam.getWidth(), rgb_cam.getHeight()
    FX_IR = (IR_W / 2.0) / math.tan(ir_left_cam.getFov() / 2.0)
    FX_RGB = (RGB_W / 2.0) / math.tan(rgb_cam.getFov() / 2.0)
    MAX_RANGE = depth_gt_sensor.getMaxRange()

    # fixed pseudo-random emitter dot pattern (~4% density, static like the real one)
    rng = np.random.default_rng(435)
    n_dots = int(0.04 * IR_W * IR_H)
    dot_x = rng.integers(1, IR_W - 1, n_dots)
    dot_y = rng.integers(1, IR_H - 1, n_dots)

    stereo_matcher = cv2.StereoSGBM_create(
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

    # ---------------------------------------------------------- kinematic state
    self_node = supervisor.getSelf()
    translation_field = self_node.getField("translation")
    rotation_field = self_node.getField("rotation")
    pos = np.array(translation_field.getSFVec3f())
    yaw = 0.0
    vel = np.zeros(3)
    prev_vel = np.zeros(3)
    yaw_rate = 0.0
    node.target_pos = pos.copy()
    node.target_yaw = yaw
    node.sim_time = supervisor.getTime()

    cam_xyz = list(param("camera_xyz"))

    def bgra(camera, width, height):
        return np.frombuffer(camera.getImage(), np.uint8).reshape((height, width, 4))

    def read_depth_gt():
        # data_type='buffer' returns a raw ctypes float pointer: wrap it
        # zero-copy, then copy out of Webots-owned memory.
        raw = depth_gt_sensor.getRangeImage(data_type="buffer")
        depth = np.ctypeslib.as_array(raw, (IR_H * IR_W,)).reshape((IR_H, IR_W)).copy()
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def emulate_ir_pair(depth_gt, noise_sigma):
        left = cv2.cvtColor(bgra(ir_left_cam, IR_W, IR_H), cv2.COLOR_BGRA2GRAY).astype(np.float32)
        right = cv2.cvtColor(bgra(ir_right_cam, IR_W, IR_H), cv2.COLOR_BGRA2GRAY).astype(np.float32)

        z = depth_gt[dot_y, dot_x]
        lit = (z > 0.1) & (z < PROJECTOR_MAX_RANGE)
        intensity = np.clip(PROJECTOR_POWER / (z * z + 1e-6), 0, 180)

        # left image: dots land where the (left-aligned) depth map says they are
        np.add.at(left, (dot_y[lit], dot_x[lit]), intensity[lit])
        # right image: same dots shifted by the true disparity fx * B / Z
        disparity = FX_IR * BASELINE / np.maximum(z, 1e-6)
        xr = np.rint(dot_x - disparity).astype(np.int32)
        ok = lit & (xr >= 0)
        np.add.at(right, (dot_y[ok], xr[ok]), intensity[ok])

        left += rng.standard_normal(left.shape, dtype=np.float32) * noise_sigma
        right += rng.standard_normal(right.shape, dtype=np.float32) * noise_sigma
        return (np.clip(left, 0, 255).astype(np.uint8),
                np.clip(right, 0, 255).astype(np.uint8))

    def stereo_depth(left_ir, right_ir):
        disparity = stereo_matcher.compute(left_ir, right_ir).astype(np.float32) / 16.0
        depth = np.zeros_like(disparity)
        valid = disparity > 0.5
        depth[valid] = FX_IR * BASELINE / disparity[valid]
        depth[depth > MAX_RANGE] = 0.0
        return depth

    print(f"D435i/ROS2 ready: RGB {RGB_W}x{RGB_H} | IR/depth {IR_W}x{IR_H}, "
          f"fx={FX_IR:.1f}px, baseline={BASELINE * 1000:.0f}mm | "
          f"command topics: /drone/cmd_pose, /drone/cmd_vel")

    last_image_time = -1.0
    last_status_time = 0.0
    depth_coverage = 0.0

    while supervisor.step(timestep) != -1:
        sim_time = supervisor.getTime()
        node.sim_time = sim_time
        rclpy.spin_once(node, timeout_sec=0.0)
        stamp = to_stamp(sim_time)
        node.pub_clock.publish(Clock(clock=stamp))

        # ---------------------------------------------------- ideal motion
        cmd_vel_fresh = (sim_time - node.last_cmd_vel_time) <= param("cmd_vel_timeout")
        if node.control_mode == "velocity" and cmd_vel_fresh:
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            vx_body, vy_body, vz_body = node.cmd_vel_linear
            vel_des = np.array([
                cos_y * vx_body - sin_y * vy_body,
                sin_y * vx_body + cos_y * vy_body,
                vz_body,
            ])
            yaw_rate_des = node.cmd_vel_yaw_rate
        elif node.control_mode == "velocity":
            # cmd_vel went stale: hover in place rather than keep flying the last command
            vel_des = np.zeros(3)
            yaw_rate_des = 0.0
        else:
            err = node.target_pos - pos
            vel_des = param("pos_gain") * err
            yaw_err = wrap_angle(node.target_yaw - yaw)
            yaw_rate_des = param("yaw_gain") * yaw_err

        xy_speed = math.hypot(vel_des[0], vel_des[1])
        if xy_speed > param("max_speed_xy"):
            vel_des[:2] *= param("max_speed_xy") / xy_speed
        vel_des[2] = max(-param("max_speed_z"), min(param("max_speed_z"), vel_des[2]))
        vel = vel_des + rng.normal(0.0, [param("motion_noise_xy"),
                                         param("motion_noise_xy"),
                                         param("motion_noise_z")])
        yaw_rate = max(-param("max_yaw_rate"),
                       min(param("max_yaw_rate"), yaw_rate_des))
        yaw_rate += rng.normal(0.0, param("motion_noise_yaw"))

        pos = pos + vel * dt
        pos[2] = max(pos[2], 0.05)  # never sink into the ground
        yaw = wrap_angle(yaw + yaw_rate * dt)
        translation_field.setSFVec3f(pos.tolist())
        rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])

        # ---------------------------------------------------- odom + tf + imu
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        quat = (0.0, 0.0, sy, cy)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = pos
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat
        # twist in base_link (yaw-only attitude)
        odom.twist.twist.linear.x = cos_y * vel[0] + sin_y * vel[1]
        odom.twist.twist.linear.y = -sin_y * vel[0] + cos_y * vel[1]
        odom.twist.twist.linear.z = vel[2]
        odom.twist.twist.angular.z = yaw_rate
        node.pub_odom.publish(odom)

        # node.pub_tf.publish(TFMessage(transforms=[
        #     make_transform(stamp, "odom", "base_link", pos.tolist(), quat),
        #     make_transform(stamp, "base_link", "d435i_link", cam_xyz, (0.0, 0.0, 0.0, 1.0)),
        #     make_transform(stamp, "d435i_link", "d435i_color_optical_frame",
        #                    [0.0125, -0.0375, 0.0], OPTICAL_QUAT),
        #     make_transform(stamp, "d435i_link", "d435i_infra1_optical_frame",
        #                    [0.0125, 0.025, 0.0], OPTICAL_QUAT),
        #     make_transform(stamp, "d435i_link", "d435i_infra2_optical_frame",
        #                    [0.0125, -0.025, 0.0], OPTICAL_QUAT),
        #     make_transform(stamp, "d435i_link", "d435i_depth_optical_frame",
        #                    [0.0125, 0.025, 0.0], OPTICAL_QUAT),
        #     make_transform(stamp, "d435i_link", "d435i_imu_frame",
        #                    [0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 1.0)),
        # ]))

        # synthesized BMI055: specific force in the body frame + yaw gyro.
        # Differentiate the noise-free commanded velocity - differentiating the
        # noise-injected one would blow white velocity noise up by 1/dt.
        accel_world = (vel_des - prev_vel) / dt + np.array([0.0, 0.0, GRAVITY])
        prev_vel = vel_des.copy()
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "d435i_imu_frame"
        imu.orientation_covariance[0] = -1.0  # no orientation, like the real device
        imu.linear_acceleration.x = (cos_y * accel_world[0] + sin_y * accel_world[1]
                                     + rng.normal(0.0, param("imu_accel_noise")))
        imu.linear_acceleration.y = (-sin_y * accel_world[0] + cos_y * accel_world[1]
                                     + rng.normal(0.0, param("imu_accel_noise")))
        imu.linear_acceleration.z = accel_world[2] + rng.normal(0.0, param("imu_accel_noise"))
        imu.angular_velocity.x = rng.normal(0.0, param("imu_gyro_noise"))
        imu.angular_velocity.y = rng.normal(0.0, param("imu_gyro_noise"))
        imu.angular_velocity.z = yaw_rate + rng.normal(0.0, param("imu_gyro_noise"))
        node.pub_imu.publish(imu)

        # ---------------------------------------------------- vision pipeline
        if sim_time - last_image_time >= vision_period_ms / 1000.0 - 1e-6:
            last_image_time = sim_time

            depth_gt = read_depth_gt()
            left_ir, right_ir = emulate_ir_pair(depth_gt, param("ir_noise_sigma"))
            rgb = cv2.cvtColor(bgra(rgb_cam, RGB_W, RGB_H), cv2.COLOR_BGRA2RGB)

            node.pub_color.publish(make_image_msg(
                stamp, "d435i_color_optical_frame", rgb, "rgb8", 3))
            node.pub_color_info.publish(make_camera_info(
                stamp, "d435i_color_optical_frame", RGB_W, RGB_H, FX_RGB))
            node.pub_infra1.publish(make_image_msg(
                stamp, "d435i_infra1_optical_frame", left_ir, "mono8", 1))
            node.pub_infra1_info.publish(make_camera_info(
                stamp, "d435i_infra1_optical_frame", IR_W, IR_H, FX_IR))
            node.pub_infra2.publish(make_image_msg(
                stamp, "d435i_infra2_optical_frame", right_ir, "mono8", 1))
            node.pub_infra2_info.publish(make_camera_info(
                stamp, "d435i_infra2_optical_frame", IR_W, IR_H, FX_IR,
                tx=-FX_IR * BASELINE))
            node.pub_depth_gt.publish(make_image_msg(
                stamp, "d435i_depth_optical_frame", depth_gt, "32FC1", 4))

            if param("enable_stereo_depth"):
                depth = stereo_depth(left_ir, right_ir)
                depth_coverage = 100.0 * np.count_nonzero(depth) / depth.size
                node.pub_depth.publish(make_image_msg(
                    stamp, "d435i_depth_optical_frame", depth, "32FC1", 4))
                node.pub_depth_info.publish(make_camera_info(
                    stamp, "d435i_depth_optical_frame", IR_W, IR_H, FX_IR))

        if sim_time - last_status_time >= 5.0:
            last_status_time = sim_time
            print(f"t={sim_time:6.1f}s  mode={node.control_mode:8s} "
                  f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:5.2f}) "
                  f"yaw={math.degrees(yaw):6.1f}deg  "
                  f"target=({node.target_pos[0]:.2f},{node.target_pos[1]:.2f},"
                  f"{node.target_pos[2]:.2f})  stereo depth coverage={depth_coverage:4.1f}%")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
