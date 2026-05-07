import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

FINGERTIP_NAMES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)

BASE_JOINT_NAMES = tuple(JOINT_NAMES)
ACTION_JOINT_NAMES = BASE_JOINT_NAMES + tuple(FINGERTIP_NAMES)
BASE_JOINT_COUNT = len(BASE_JOINT_NAMES)
FINGERTIP_COUNT = len(FINGERTIP_NAMES)
ACTION_JOINT_COUNT = len(ACTION_JOINT_NAMES)
ACTION_STATE_DIM = 3
ACTION_STATE_DELTA_FPS_SCALE = 24.0

JOINT_NAME_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_JOINT_NAMES)}

# Explicit bone list for the 65-joint action order above.
# Keeping this as joint-name pairs makes downstream loss code easier to read.
SMPLX_BONE_NAME_EDGES = (
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("pelvis", "spine1"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("spine1", "spine2"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("spine2", "spine3"),
    ("left_ankle", "left_foot"),
    ("right_ankle", "right_foot"),
    ("spine3", "neck"),
    ("spine3", "left_collar"),
    ("spine3", "right_collar"),
    ("neck", "head"),
    ("left_collar", "left_shoulder"),
    ("right_collar", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("head", "jaw"),
    ("head", "left_eye_smplhf"),
    ("head", "right_eye_smplhf"),
    ("left_wrist", "left_index1"),
    ("left_index1", "left_index2"),
    ("left_index2", "left_index3"),
    ("left_wrist", "left_middle1"),
    ("left_middle1", "left_middle2"),
    ("left_middle2", "left_middle3"),
    ("left_wrist", "left_pinky1"),
    ("left_pinky1", "left_pinky2"),
    ("left_pinky2", "left_pinky3"),
    ("left_wrist", "left_ring1"),
    ("left_ring1", "left_ring2"),
    ("left_ring2", "left_ring3"),
    ("left_wrist", "left_thumb1"),
    ("left_thumb1", "left_thumb2"),
    ("left_thumb2", "left_thumb3"),
    ("right_wrist", "right_index1"),
    ("right_index1", "right_index2"),
    ("right_index2", "right_index3"),
    ("right_wrist", "right_middle1"),
    ("right_middle1", "right_middle2"),
    ("right_middle2", "right_middle3"),
    ("right_wrist", "right_pinky1"),
    ("right_pinky1", "right_pinky2"),
    ("right_pinky2", "right_pinky3"),
    ("right_wrist", "right_ring1"),
    ("right_ring1", "right_ring2"),
    ("right_ring2", "right_ring3"),
    ("right_wrist", "right_thumb1"),
    ("right_thumb1", "right_thumb2"),
    ("right_thumb2", "right_thumb3"),
    ("left_thumb3", "left_thumb"),
    ("left_index3", "left_index"),
    ("left_middle3", "left_middle"),
    ("left_ring3", "left_ring"),
    ("left_pinky3", "left_pinky"),
    ("right_thumb3", "right_thumb"),
    ("right_index3", "right_index"),
    ("right_middle3", "right_middle"),
    ("right_ring3", "right_ring"),
    ("right_pinky3", "right_pinky"),
)

SMPLX_BONE_EDGES = tuple(
    (JOINT_NAME_TO_INDEX[parent_name], JOINT_NAME_TO_INDEX[child_name])
    for parent_name, child_name in SMPLX_BONE_NAME_EDGES
)

SMPLX_WRIST_TO_FINGER_ROOT_BONE_NAME_EDGES = (
    ("left_wrist", "left_thumb1"),
    ("left_wrist", "left_index1"),
    ("left_wrist", "left_middle1"),
    ("left_wrist", "left_ring1"),
    ("left_wrist", "left_pinky1"),
    ("right_wrist", "right_thumb1"),
    ("right_wrist", "right_index1"),
    ("right_wrist", "right_middle1"),
    ("right_wrist", "right_ring1"),
    ("right_wrist", "right_pinky1"),
)
SMPLX_WRIST_TO_FINGER_ROOT_BONE_EDGES = tuple(
    (JOINT_NAME_TO_INDEX[parent_name], JOINT_NAME_TO_INDEX[child_name])
    for parent_name, child_name in SMPLX_WRIST_TO_FINGER_ROOT_BONE_NAME_EDGES
)
_SMPLX_WRIST_TO_FINGER_ROOT_BONE_EDGES = set(SMPLX_WRIST_TO_FINGER_ROOT_BONE_EDGES)
SMPLX_WRIST_TO_FINGER_ROOT_BONE_MASK = tuple(
    edge in _SMPLX_WRIST_TO_FINGER_ROOT_BONE_EDGES for edge in SMPLX_BONE_EDGES
)

# ---- Rotation conversions (axis-angle <-> 6D) ----
def axis_angle_to_rot6(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: [..., 3] (Rodrigues / rotvec)
    returns:    [..., 6] (first two columns of rotation matrix, flattened col-wise)
    """
    aa = np.asarray(axis_angle, dtype=np.float64)
    orig_shape = aa.shape[:-1]
    aa_flat = aa.reshape(-1, 3)

    mats = R.from_rotvec(aa_flat).as_matrix()  # [N, 3, 3]
    # Take first two columns (3x2) and flatten. We'll flatten column-wise for clarity:
    # col0 (3), col1 (3) -> 6
    rot6 = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=1)  # [N, 6]
    return rot6.reshape(*orig_shape, 6)


def rot6_to_matrix(rot6: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    rot6: [..., 6] -> rotation matrices [..., 3, 3]
    Uses Gram-Schmidt as in the 6D rep paper practice.
    """
    x = np.asarray(rot6, dtype=np.float64)
    orig_shape = x.shape[:-1]
    x = x.reshape(-1, 6)

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    def normalize(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, eps, None)

    b1 = normalize(a1)
    # Make a2 orthogonal to b1
    proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = normalize(a2 - proj)
    b3 = np.cross(b1, b2)

    mats = np.stack([b1, b2, b3], axis=2)  # [N, 3, 3] columns are b1,b2,b3
    return mats.reshape(*orig_shape, 3, 3)


def rot6_to_axis_angle(rot6: np.ndarray) -> np.ndarray:
    """
    rot6: [..., 6] -> axis-angle [..., 3]
    """
    mats = rot6_to_matrix(rot6)  # [...,3,3]
    orig_shape = mats.shape[:-2]
    mats_flat = mats.reshape(-1, 3, 3)
    aa = R.from_matrix(mats_flat).as_rotvec()  # [N,3]
    return aa.reshape(*orig_shape, 3)


# ---- Field <-> joint index mapping for the 65-joint action order ----
# Indices are in ACTION_JOINT_NAMES order.
_PELVIS = 0
_LWRIST = 20
_RWRIST = 21
_JAW = 22
_LEYE = 23
_REYE = 24

_BODY_POSE_IDXS = list(range(1, 22))  # 21 joints: left_hip..right_wrist
_FINGERTIP_START = BASE_JOINT_COUNT
_LFINGERTIP_IDXS = list(range(_FINGERTIP_START, _FINGERTIP_START + 5))
_RFINGERTIP_IDXS = list(range(_FINGERTIP_START + 5, _FINGERTIP_START + FINGERTIP_COUNT))
_LHAND_IDXS = list(range(25, 40)) + _LFINGERTIP_IDXS  # 20 joints: left fingers + fingertips
_RHAND_IDXS = list(range(40, 55)) + _RFINGERTIP_IDXS  # 20 joints: right fingers + fingertips
_HAND_POSITION_SCALE = 5.0


def _ensure_frames_first(x: np.ndarray, n_frames: int, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[0] != n_frames:
        raise ValueError(f"{name}: expected first dim n_frames={n_frames}, got {x.shape}")
    return x


def _as_frames_joints3(x: np.ndarray, n_frames: int, expected_joints: int | None, name: str) -> np.ndarray:
    """
    Accepts either [T,3] or [T,J,3]. Returns [T,J,3] if expected_joints is not None else [T,1,3].
    """
    x = _ensure_frames_first(x, n_frames, name)
    if x.ndim == 2 and x.shape[1] == 3:
        x = x[:, None, :]  # [T,1,3]
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"{name}: expected shape [T,3] or [T,J,3], got {x.shape}")
    if expected_joints is not None and x.shape[1] != expected_joints:
        raise ValueError(f"{name}: expected {expected_joints} joints, got {x.shape[1]} (shape {x.shape})")
    return x


def normalize_action_state_positions(action_states: np.ndarray) -> np.ndarray:
    x = np.asarray(action_states, dtype=np.float64).copy()
    if x.ndim != 3 or x.shape[1:] != (ACTION_JOINT_COUNT, ACTION_STATE_DIM):
        raise ValueError(
            f"action_states expected [T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {x.shape}"
        )

    pos_abs = x[:, :, 0:3].copy()
    pelvis_traj = pos_abs[:, _PELVIS : _PELVIS + 1, :]
    pelvis_0 = pelvis_traj[0:1, :, :]

    x[:, :, 0:3] = pos_abs - pelvis_traj
    x[:, _LHAND_IDXS, 0:3] = pos_abs[:, _LHAND_IDXS, :] - pos_abs[:, _LWRIST : _LWRIST + 1, :]
    x[:, _RHAND_IDXS, 0:3] = pos_abs[:, _RHAND_IDXS, :] - pos_abs[:, _RWRIST : _RWRIST + 1, :]
    x[:, _LHAND_IDXS, 0:3] *= _HAND_POSITION_SCALE
    x[:, _RHAND_IDXS, 0:3] *= _HAND_POSITION_SCALE
    x[:, _PELVIS, 0:3] = pelvis_traj[:, 0, :] - pelvis_0[0, 0, :]
    return x


def recover_absolute_action_state_positions(action_states: np.ndarray, pelvis_0: np.ndarray) -> np.ndarray:
    x = np.asarray(action_states, dtype=np.float64).copy()
    if x.ndim != 3 or x.shape[1:] != (ACTION_JOINT_COUNT, ACTION_STATE_DIM):
        raise ValueError(
            f"action_states expected [T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {x.shape}"
        )

    pelvis_0 = np.asarray(pelvis_0, dtype=np.float64)
    if pelvis_0.shape != (3,):
        raise ValueError(f"pelvis_0 expected [3], got {pelvis_0.shape}")

    pos_rel = x[:, :, 0:3]
    pelvis_traj = pos_rel[:, _PELVIS, :] + pelvis_0[None, :]
    pos_abs = pos_rel + pelvis_traj[:, None, :]
    pos_abs[:, _PELVIS, :] = pelvis_traj

    left_wrist = pos_abs[:, _LWRIST : _LWRIST + 1, :].copy()
    right_wrist = pos_abs[:, _RWRIST : _RWRIST + 1, :].copy()
    pos_abs[:, _LHAND_IDXS, :] = pos_rel[:, _LHAND_IDXS, :] / _HAND_POSITION_SCALE + left_wrist
    pos_abs[:, _RHAND_IDXS, :] = pos_rel[:, _RHAND_IDXS, :] / _HAND_POSITION_SCALE + right_wrist

    x[:, :, 0:3] = pos_abs
    return x


def recover_absolute_action_state_positions_tensor(
    action_states: torch.Tensor,
    pelvis_0: torch.Tensor,
) -> torch.Tensor:
    if action_states.ndim != 4 or tuple(action_states.shape[-2:]) != (ACTION_JOINT_COUNT, ACTION_STATE_DIM):
        raise ValueError(
            f"action_states expected [B,T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {tuple(action_states.shape)}"
        )

    if pelvis_0.ndim == 1:
        pelvis_0 = pelvis_0.unsqueeze(0)
    if pelvis_0.ndim != 2 or pelvis_0.shape[1] != 3:
        raise ValueError(f"pelvis_0 expected [3] or [B,3], got {tuple(pelvis_0.shape)}")
    if pelvis_0.shape[0] not in (1, action_states.shape[0]):
        raise ValueError(
            f"pelvis_0 batch mismatch: expected 1 or {action_states.shape[0]}, got {pelvis_0.shape[0]}"
        )

    x = action_states.clone()
    pelvis_0 = pelvis_0.to(device=x.device, dtype=x.dtype)

    pos_rel = x[..., 0:3]
    pelvis_traj = pos_rel[:, :, _PELVIS, :] + pelvis_0[:, None, :]
    pos_abs = pos_rel + pelvis_traj[:, :, None, :]
    pos_abs[:, :, _PELVIS, :] = pelvis_traj

    left_wrist = pos_abs[:, :, _LWRIST : _LWRIST + 1, :].clone()
    right_wrist = pos_abs[:, :, _RWRIST : _RWRIST + 1, :].clone()
    pos_abs[:, :, _LHAND_IDXS, :] = pos_rel[:, :, _LHAND_IDXS, :] / _HAND_POSITION_SCALE + left_wrist
    pos_abs[:, :, _RHAND_IDXS, :] = pos_rel[:, :, _RHAND_IDXS, :] / _HAND_POSITION_SCALE + right_wrist

    x[..., 0:3] = pos_abs
    return x


# ---- Pack: SMPL-X dict -> [T,65,3] ----
def smplx_dict_to_action_states(seq_data: dict) -> np.ndarray:
    """
    Returns action_states [T,65,3] containing joint positions only.
    """
    joints3d = np.asarray(seq_data["joints3d"], dtype=np.float64)  # [T, 55, 3]
    if joints3d.ndim != 3 or joints3d.shape[1:] != (BASE_JOINT_COUNT, 3):
        raise ValueError(f"joints3d expected [T,{BASE_JOINT_COUNT},3], got {joints3d.shape}")

    fingertips = np.asarray(seq_data["joints3d_fingertips"], dtype=np.float64)  # [T, 10, 3]
    if fingertips.ndim != 3 or fingertips.shape != (joints3d.shape[0], FINGERTIP_COUNT, 3):
        raise ValueError(f"joints3d_fingertips expected [T,{FINGERTIP_COUNT},3], got {fingertips.shape}")

    T = joints3d.shape[0]  # number of frames
    joints3d_full = np.concatenate([joints3d, fingertips], axis=1)  # [T, 65, 3]
    out = np.zeros((T, ACTION_JOINT_COUNT, ACTION_STATE_DIM), dtype=np.float64)
    out[:, :, 0:3] = joints3d_full
    return out


def action_states_to_smplx_dict(action_states: np.ndarray, betas: np.ndarray) -> dict:
    """
    action_states: [T,65,3] containing joint positions only.
    Pose fields are returned as zeros.
    """
    x = np.asarray(action_states, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (ACTION_JOINT_COUNT, ACTION_STATE_DIM):
        raise ValueError(
            f"action_states expected [T,{ACTION_JOINT_COUNT},{ACTION_STATE_DIM}], got {x.shape}"
        )

    T = x.shape[0]
    joints3d = x[:, :BASE_JOINT_COUNT, 0:3]
    joints3d_fingertips = x[:, _FINGERTIP_START:, 0:3]
    betas = np.asarray(betas, dtype=np.float64)
    if betas.shape != (T, 10):
        raise ValueError(f"betas expected [T,10], got {betas.shape}")

    zeros_pose = np.zeros((T, 3), dtype=np.float64)
    zeros_body = np.zeros((T, 21, 3), dtype=np.float64)
    zeros_hand = np.zeros((T, 15, 3), dtype=np.float64)

    out = {
        "transl": joints3d[:, _PELVIS, :],
        "global_orient": zeros_pose,
        "body_pose": zeros_body,
        "jaw_pose": zeros_pose,
        "leye_pose": zeros_pose,
        "reye_pose": zeros_pose,
        "left_hand_pose": zeros_hand,
        "right_hand_pose": zeros_hand,
        "betas": betas,
        "joints3d": joints3d,
        "joints3d_fingertips": joints3d_fingertips,
    }
    return out
