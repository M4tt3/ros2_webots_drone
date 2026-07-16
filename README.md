# RealSense D435i drone simulation (ROS 2)

A Webots (R2025a) simulation of an ideal (kinematic) drone on grass carrying a
**fully emulated Intel RealSense D435i**: RGB module, both IR imagers with the
projected dot pattern, stereo-matched depth, and a synthesized IMU. Everything
is published to ROS 2 topics; the drone executes x/y/z/yaw pose setpoints
"ideally" with velocity-limited, parameter-tunable noisy motion.

## Run it

```bash
# from a shell with ROS 2 sourced (source /opt/ros/humble/setup.bash)
webots worlds/d435i_grass.wbt
```

Then command the drone (position in metres, yaw as quaternion) — this switches it to position-hold mode:

```bash
ros2 topic pub --once /drone/cmd_pose geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 3.0, y: 1.0, z: 2.0}, orientation: {z: 0.7071, w: 0.7071}}}"
```

Or drive it with a body-frame velocity setpoint — this switches it to velocity mode, and it
hovers in place if the topic goes quiet for `cmd_vel_timeout` seconds:

```bash
ros2 topic pub -r 20 /drone/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.2}}"
```

View the streams with `rviz2` or `ros2 run rqt_image_view rqt_image_view`.

## Topics

| Topic | Type | Content |
| --- | --- | --- |
| `/drone/cmd_pose` (sub) | `geometry_msgs/PoseStamped` | x/y/z + yaw setpoint; switches to position-hold mode |
| `/drone/cmd_vel` (sub) | `geometry_msgs/Twist` | body-frame (FLU) linear velocity + yaw rate setpoint; switches to velocity mode, hovers if stale |
| `/drone/odom` | `nav_msgs/Odometry` | ground-truth pose + body twist, ~125 Hz |
| `/d435i/color/image_raw` + `camera_info` | `sensor_msgs/Image` | rgb8, 640x360, 69° HFOV |
| `/d435i/infra1/image_raw` + `camera_info` | `sensor_msgs/Image` | mono8, left IR + dots + noise |
| `/d435i/infra2/image_raw` + `camera_info` | `sensor_msgs/Image` | mono8, right IR (P matrix carries the -fx·B baseline) |
| `/d435i/depth/image_rect_raw` + `camera_info` | `sensor_msgs/Image` | 32FC1 metres, **SGBM on the IR pair** (0 = invalid) |
| `/d435i/depth_gt/image_raw` | `sensor_msgs/Image` | 32FC1 metres, RangeFinder ground truth |
| `/d435i/imu` | `sensor_msgs/Imu` | synthesized BMI055 accel + gyro, ~125 Hz |
| `/tf`, `/clock` | | odom→base_link→d435i_link→optical frames; sim time |

## How the D435i is emulated

`protos/RealSenseD435i.proto` models the real device (90 x 25 x 25 mm body,
50 mm baseline, sensors at their true positions, real FOVs); the pipeline in
`controllers/ideal_drone_ros2/ideal_drone_ros2.py` does the rest at ~15 Hz:

1. The two 87° IR renders are converted to 8-bit mono (the OV9282s are monochrome).
2. The IR emitter's fixed pseudo-random dot pattern is projected into **both**
   images: each dot lands at its true position in the left image and is shifted
   by the true disparity `fx·B/Z` in the right one, with 1/Z² intensity falloff
   and an 8 m reach — geometrically consistent active stereo.
3. Gaussian read noise is added to both IR images.
4. Depth is computed like the D4 ASIC: semi-global matching (OpenCV SGBM) on
   the noisy IR pair, `depth = fx·B/disparity`. The RangeFinder is only used to
   drive the projector and as the `depth_gt` reference — so the published depth
   shows realistic failure modes (no return on sky, washed-out dots at range,
   occlusion shadows, noise speckle).
5. The IMU is synthesized from the drone kinematics: body-frame specific force
   (including gravity) + yaw-rate gyro, with configurable noise.

## Ideal drone

`protos/IdealDrone.proto` has no physics and no propellers — the controller is
a Supervisor that moves the robot toward the setpoint each 8 ms step, velocity
limits and optional white velocity noise applied either way. In position-hold
mode it runs a P-controller on position/yaw; in velocity mode it integrates
the commanded body-frame velocity/yaw-rate directly (falling back to hover if
`/drone/cmd_vel` goes stale). Commands are therefore executed exactly, plus
whatever noise you dial in.

## Parameters

ROS parameters on the `d435i_drone` node — change at runtime with
`ros2 param set /d435i_drone <name> <value>` or permanently via `controllerArgs`
in the world file (e.g. `controllerArgs ["--ros-args", "-p", "max_speed_xy:=5.0"]`):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_speed_xy` / `max_speed_z` | 2.0 / 1.5 m/s | velocity limits |
| `max_yaw_rate` | 1.5 rad/s | yaw rate limit |
| `pos_gain` / `yaw_gain` | 1.5 / 2.0 1/s | P gains (approach responsiveness, position-hold mode only) |
| `cmd_vel_timeout` | 0.5 s | hover if `/drone/cmd_vel` goes stale for this long |
| `motion_noise_xy` / `_z` / `_yaw` | 0.03 / 0.02 m/s, 0.01 rad/s | white noise injected into the executed velocity (0 = perfectly ideal) |
| `imu_accel_noise` / `imu_gyro_noise` | 0.05 m/s², 0.005 rad/s | IMU noise |
| `ir_noise_sigma` | 4.0 grey levels | IR image read noise |
| `image_rate` | 15 Hz | camera/depth publish rate (read at startup) |
| `enable_stereo_depth` | true | run SGBM and publish `/d435i/depth` |
| `camera_xyz` | [0.06, 0, -0.005] | D435i mount offset — keep in sync with the world's `sensorSlot` translation |

Camera resolutions are proto fields on the `RealSenseD435i` node in the world
(`rgbWidth/rgbHeight`, `depthWidth/depthHeight`, `maxRange`).

## Known limitations

- Webots renders the visible spectrum, so IR reflectance is approximated by
  grayscale visible-light reflectance.
- Projected dots are not occlusion-tested in the right image.
- SGBM's 128 disparities bound the minimum stereo depth to ~0.17 m at 848 px
  (the real min-Z is 0.105 m); raise `numDisparities` in the controller if needed.
- The drone has no collision response (kinematic): it will fly through obstacles.

## Requirements

- Webots R2025a (assets fetched from the official URLs on first load, then cached)
- ROS 2 (tested with Humble) and Python 3 with `numpy` + `opencv-python`
- Launch Webots from a ROS-sourced shell so the controller can import `rclpy`
