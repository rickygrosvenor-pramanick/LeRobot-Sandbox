# Fine-tuning Isaac-GR00T for a UR5e Arm with Robotiq Gripper

This guide provides an in-depth walkthrough for fine-tuning the NVIDIA Isaac-GR00T foundation model for a UR5e robotic arm equipped with a Robotiq Gripper and a Zed Camera for vision. It is specifically tailored for users with an RTX 4090 GPU.

## 🎯 Overview: The Goal

Our goal is to collect high-quality demonstration data from your UR5e setup and convert it into the `LeRobot-compatible` format required by Isaac-GR00T. This will enable you to fine-tune the model to perform manipulation tasks with your specific hardware.

Given your setup (UR5e arm, TCP data collection) and GPU (RTX 4090), we will be targeting the **`EmbodimentTag.OXE_DROID`** action head. This head is optimized for single-arm robots using **delta end-effector (EEF) control**, which aligns perfectly with your TCP data collection.

## 📊 Data You Need to Collect

The quality of your fine-tuned model is directly dependent on the quality and diversity of your collected data. For each demonstration, you need to record synchronized streams of state, action, and video data.

### 1. State Data (`observation.state`)

This is a snapshot of your robot's state at each timestep. It should be a flat array of numbers. For your UR5e setup, we recommend the following components in this specific order:

| Component | Description | Dimensions | Notes |
| :--- | :--- | :--- | :--- |
| **Joint Positions** | The angles of the 6 UR5e joints. | 6 | In radians. Essential for the model to understand the robot's configuration. |
| **Joint Velocities** | The angular velocities of the 6 UR5e joints. | 6 | In radians/sec. Provides dynamic information. (Optional but recommended) |
| **Gripper State** | The position/openness of the Robotiq Gripper. | 1-2 | A single value for openness (0.0-1.0) or two values for finger positions. |
| **TCP Pose (Absolute)** | The absolute pose of your tool center point. | 7 | 3 for position (x, y, z) and 4 for orientation (quaternion: x, y, z, w). |

**Total State Dimension**: 6 + 6 + 1 + 7 = **20 dimensions** (if all components are used).

### 2. Action Data (`action`)

This represents the command sent to the robot at each timestep. Since we are using the `OXE_DROID` head, your actions should be in the **delta end-effector space**.

| Component | Description | Dimensions | Notes |
| :--- | :--- | :--- | :--- |
| **Delta TCP Position**| The change in TCP position from the previous step. | 3 | `delta_pos = pos_{t+1} - pos_t` |
| **Delta TCP Orientation**| The change in TCP orientation. | 3 | Axis-angle representation is common. `delta_rot = rot_{t+1} * inverse(rot_t)` |
| **Gripper Command** | The target state for the gripper. | 1 | Typically a value from 0.0 (closed) to 1.0 (open). |

**Total Action Dimension**: 3 + 3 + 1 = **7 dimensions**.

**How to Calculate Delta Actions from your TCP data:**
You are already collecting absolute TCP poses. You can derive the required delta actions during your data processing step:
- **Delta Position**: `delta_pos_t = tcp_pos_{t+1} - tcp_pos_t`
- **Delta Orientation**: This is a rotation from the orientation at `t` to `t+1`. You can calculate this using quaternion multiplication: `delta_quat = quat_{t+1} * inverse(quat_t)`. Then, convert this `delta_quat` to a 3D axis-angle representation for the action vector.

### 3. Video Data (`observation.images.*`)

This is the visual input from your Zed Camera.

- **Format**: MP4
- **Resolution**: A resolution of `480x640` is standard and works well. Higher resolutions increase data size and processing load.
- **Frame Rate (FPS)**: Aim for a consistent frame rate, e.g., **20-30 FPS**. This should be synchronized with your state and action data collection frequency.
- **Camera Placement**: Mount the camera in a fixed position that provides a clear, unobstructed view of the robot's workspace. A "third-person" view is typical. If you have multiple cameras (e.g., wrist and static), you can include both streams.

### 4. Language Annotations (`annotation.*`)

This is crucial for teaching the model to associate language commands with actions. For each demonstration episode, write a clear, concise description of the task.

- **Examples**:
  - `"pick up the red block and place it in the green bowl"`
  - `"open the drawer"`
  - `"pour water from the bottle into the cup"`
- **Storage**: These will be stored in the `meta/tasks.jsonl` file.

### 5. Data Collection Frequency

A critical aspect of collecting high-quality data is the frequency at which you record your trajectories.

-   **Recommended Frequency**: A good range for most manipulation tasks is **10-30 Hz**. The existing recommendation of 20-30 FPS for video data aligns with this. Your state and action data must be collected at this same, consistent frequency.
-   **Consistency is Key**: Avoid variable frequencies. A consistent time step between data points helps the model learn the system's dynamics.
-   **Synchronization**: All data streams—state, action, and video frames—must be tightly synchronized. Each timestamp in your state and action data should correspond to a specific video frame. Misaligned data can severely degrade model performance.

## 🗂️ Structuring Your Data for Conversion

Once you have collected the raw data, you need to process it and structure it for the conversion scripts. The `data_conversion` module in this repository is designed to handle this.

### Your `modality.json` for UR5e

This file is the most critical piece of metadata. It tells the data loader how to interpret your flat state and action arrays. For your setup, it should look like this:

```json
{
  "state": {
    "joint_positions": { "start": 0, "end": 6 },
    "joint_velocities": { "start": 6, "end": 12 },
    "gripper_state": { "start": 12, "end": 13 },
    "tcp_pose": { "start": 13, "end": 20 }
  },
  "action": {
    "delta_tcp_position": { "start": 0, "end": 3, "absolute": false },
    "delta_tcp_orientation": { "start": 3, "end": 6, "absolute": false },
    "gripper_command": { "start": 6, "end": 7, "absolute": true }
  },
  "video": {
    "zed_camera": {
      "original_key": "observation.images.zed_camera"
    }
  },
  "annotation": {
    "human.action.task_description": {},
    "human.validity": {}
  }
}
```
**Note**: The `"absolute": false` flag for delta actions is important. It tells the model that these are relative changes, not absolute targets.

### Your `parquet` File Content

Each row in your `episode_*.parquet` file will represent one timestep and should contain:
- `observation.state`: A 20-element list/array (e.g., `[jp1, ..., jp6, jv1, ..., jv6, gs, tcpx, ..., tcpw]`).
- `action`: A 7-element list/array (e.g., `[dx, dy, dz, dax, day, daz, gc]`).
- `timestamp`: The timestamp of the observation.
- `annotation.human.action.task_description`: An integer index pointing to the task description in `meta/tasks.jsonl`.

## 🖥️ Fine-tuning on an RTX 4090

The RTX 4090 is a powerful GPU, but fine-tuning large foundation models can be memory-intensive. To ensure a smooth process, follow these recommendations:

1.  **Use LoRA Fine-tuning**: Low-Rank Adaptation (LoRA) is a memory-efficient technique that freezes the main model weights and trains only small adapter layers. This dramatically reduces memory usage.
2.  **Disable Diffusion Model Tuning**: The diffusion model (the action head) is a significant source of memory consumption. The pre-trained head is often sufficient. You can disable tuning it to save VRAM.
3.  **Use Mixed-Precision Training**: Use `float16` or `bfloat16` to reduce memory footprint.
4.  **Gradient Accumulation**: Use a small batch size (e.g., 1 or 2) and compensate with a larger number of gradient accumulation steps to achieve a larger effective batch size.

I have already provided a **`consumer_gpu` configuration** that is optimized for this scenario. You can use it as a starting point.

## 🚀 Workflow Summary

1.  **Collect Data**: Record synchronized streams of joint states, TCP poses, gripper states, and Zed camera video for various manipulation tasks.
2.  **Write Language Annotations**: For each task, write a clear text description.
3.  **Process Data**:
    - Convert your raw data into per-episode files (e.g., pickle, HDF5).
    - Calculate the delta actions from your absolute TCP poses.
    - Ensure all data is synchronized and at a consistent frequency.
4.  **Convert to LeRobot Format**:
    - Configure your embodiment and topics in a script (see `examples/convert_data_example.py`).
    - Use the provided converters (`RosbagConverter`, `HDF5Converter`, or a custom one) to generate the LeRobot dataset.
5.  **Validate Dataset**:
    - Run `python scripts/validate_dataset.py --dataset-path /path/to/your/dataset` to ensure it's correctly formatted.
6.  **Fine-tune**:
    - Use the `consumer_gpu` configuration as a template.
    - Launch the fine-tuning script, making sure to use the `--no-tune_diffusion_model` flag.
7.  **Evaluate and Deploy**: Test your fine-tuned model and integrate it into your robot control stack.

This detailed guide should provide a clear path for you to successfully fine-tune Isaac-GR00T on your UR5e robot.
