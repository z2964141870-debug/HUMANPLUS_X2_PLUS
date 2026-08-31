import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from Aplus.tools.smpl_light import *
from articulate.math import *

# body_model = SMPLight()
class SMPLXConverter:
    """将24关节动捕数据转换为SMPL-X格式"""

    SMPL_JOINT_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
    ]
    
    def __init__(self):
        # SMPL-X关节名称（24关节版本）
        # self.smplx_joint_names = [
        #     "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
        #     "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        #     "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        #     "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
        # ]
        self.smplx_dic = ["trans", "betas", "root_orient","pose_body"]
        # 关节层级关系（父关节索引）
        self.parent_indices = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21
        ]
        self.body_model = SMPLight()
        self._cache = {}
        self._joint_names_cache = None
        self._parents_cache = None
        self._betas_np = None  # 已废弃: betas 维度由 _get_cached_tensors 按 body_model.num_betas 动态生成
        self._fast_cache = {}

    

    def _get_model_device(self, body_model):
        try:
            return next(body_model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    def _get_cached_tensors(self, body_model, num_frames: int):
        device = self._get_model_device(body_model)
        key = (str(device), num_frames)
        cached = self._cache.get(key)
        if cached is None:
            cached = {
                "betas": torch.zeros((1, body_model.num_betas), dtype=torch.float32, device=device),
                "left_hand_pose": torch.zeros((num_frames, 45), dtype=torch.float32, device=device),
                "right_hand_pose": torch.zeros((num_frames, 45), dtype=torch.float32, device=device),
                "jaw_pose": torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
                "leye_pose": torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
                "reye_pose": torch.zeros((num_frames, 3), dtype=torch.float32, device=device),
                "T": torch.tensor(
                    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                    dtype=torch.float32,
                    device=device,
                ),
            }
            self._cache[key] = cached
        return cached, device

    def _get_joint_metadata(self, body_model):
        if self._joint_names_cache is None or self._parents_cache is None:
            from smplx.joint_names import JOINT_NAMES

            self._joint_names_cache = JOINT_NAMES[: len(body_model.parents)]
            self._parents_cache = body_model.parents
        return self._joint_names_cache, self._parents_cache

    def _run_smplx_model(self, body_model, trans_np, root_orient_np, pose_body_np):
        num_frames = pose_body_np.shape[0]
        cached, device = self._get_cached_tensors(body_model, num_frames)

        trans = torch.as_tensor(trans_np, dtype=torch.float32, device=device)
        global_orient = torch.as_tensor(root_orient_np.reshape(num_frames, 3), dtype=torch.float32, device=device)
        body_pose = torch.as_tensor(pose_body_np.reshape(num_frames, 63), dtype=torch.float32, device=device)

        with torch.inference_mode():
            smplx_output = body_model(
                betas=cached["betas"],
                global_orient=global_orient,
                body_pose=body_pose,
                transl=trans,
                left_hand_pose=cached["left_hand_pose"],
                right_hand_pose=cached["right_hand_pose"],
                jaw_pose=cached["jaw_pose"],
                leye_pose=cached["leye_pose"],
                reye_pose=cached["reye_pose"],
                return_full_pose=True,
            )
        return smplx_output, cached["T"], device

    def _smplx_output_to_frames(self, body_model, smplx_output):
        curr_frame = 0
        global_orient = smplx_output.global_orient[curr_frame].detach().cpu().numpy().squeeze()
        full_body_pose = smplx_output.full_pose[curr_frame].detach().cpu().numpy().reshape(-1, 3)
        joints = smplx_output.joints[curr_frame].detach().cpu().numpy().squeeze()
        joint_names, parents = self._get_joint_metadata(body_model)

        result = {}
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(full_body_pose[i].squeeze())
            joint_orientations.append(rot)
            result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))
        return result

    def _prepare_smpl_inputs(self, axis_angles, root_translation, body_model):
        cached, device = self._get_cached_tensors(body_model, 1)
        axis_angles_tensor = torch.as_tensor(axis_angles, dtype=torch.float32, device=device)
        rotation_matrices = self.axis_angle_to_rotation_matrix(axis_angles_tensor)
        T = cached["T"].to(dtype=rotation_matrices.dtype)

        rotation_matrices_new = T @ rotation_matrices[0]
        rotation_matrices[0, ...] = rotation_matrices_new
        root_orient_aa = rotation_matrix_to_axis_angle(rotation_matrices_new)
        body_axis_angles = rotation_matrix_to_axis_angle(rotation_matrices)

        trans_np = np.asarray(root_translation, dtype=np.float32).copy()
        trans_np[[0, 1, 2]] = trans_np[[0, 2, 1]]
        root_orient_np = root_orient_aa.detach().cpu().numpy().reshape(1, 1, 3).astype(np.float32)
        pose_body_np = body_axis_angles.flatten(0)[3:66].detach().cpu().numpy().reshape(1, 63).astype(np.float32)
        smpl_pose_np = pose_body_np.reshape(1, 21, 3)
        return trans_np, root_orient_np, pose_body_np, smpl_pose_np

    def _get_fast_cached_tensors(self, device):
        key = str(device)
        cached = self._fast_cache.get(key)
        if cached is None:
            cached = {
                "T": torch.tensor(
                    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                    dtype=torch.float32,
                    device=device,
                ),
            }
            self._fast_cache[key] = cached
        return cached

    def convert_axis_angle_to_human_data_fast(
        self, axis_angles, root_translation, device="auto"
    ):
        """Build only the 24-joint GMR input using lightweight SMPL FK."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        cached = self._get_fast_cached_tensors(device)

        axis_angles_tensor = torch.as_tensor(
            axis_angles, dtype=torch.float32, device=device
        ).reshape(24, 3)
        rotation_matrices = self.axis_angle_to_rotation_matrix(axis_angles_tensor)
        rotation_matrices[0] = cached["T"] @ rotation_matrices[0]

        root_translation_tensor = torch.as_tensor(
            root_translation, dtype=torch.float32, device=device
        ).reshape(3)[[0, 2, 1]]
        with torch.inference_mode():
            joint_rot, joint_pos = self.body_model.forward_kinematics(
                rotation_matrices.unsqueeze(0),
                trans=root_translation_tensor.reshape(1, 3),
                calc_joint=True,
            )

        joint_rot_np = joint_rot[0].detach().cpu().numpy()
        joint_pos_np = joint_pos[0].detach().cpu().numpy().astype(np.float32)
        quat_xyzw = R.from_matrix(joint_rot_np).as_quat().astype(np.float32)
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        return {
            name: (joint_pos_np[i], quat_wxyz[i])
            for i, name in enumerate(self.SMPL_JOINT_NAMES)
        }

    def convert_axis_angle_to_smplx_bundle(self, axis_angles, root_translation, body_model):
        trans_np, root_orient_np, pose_body_np, smpl_pose_np = self._prepare_smpl_inputs(
            axis_angles, root_translation, body_model
        )
        smplx_output, _, _ = self._run_smplx_model(
            body_model=body_model,
            trans_np=trans_np.reshape(1, 3),
            root_orient_np=root_orient_np,
            pose_body_np=pose_body_np,
        )
        human_data = self._smplx_output_to_frames(body_model, smplx_output)
        smpl_joints_np = (
            smplx_output.joints[0, :24, :].detach().cpu().numpy().astype(np.float32).reshape(1, 24, 3)
        )
        return {
            "human_data": human_data,
            "smpl_joints": smpl_joints_np,
            "smpl_pose": smpl_pose_np.astype(np.float32),
        }

    def convert_axis_angle_to_smplx(self, axis_angles, root_translation, body_model):
        """
        将轴角表示转换为SMPL-X格式
        
        Args:
            axis_angles: [24, 3] 24个关节的轴角表示
            root_translation: [3,] 根关节平移
            
        Returns:
            smplx_data: 字典，键为SMPL-X身体部位名称，值为[位置, 四元数]
        """
        return self.convert_axis_angle_to_smplx_bundle(
            axis_angles, root_translation, body_model
        )["human_data"]
    
    def axis_angle_to_rotation_matrix(self, axis_angles):
        """将轴角转换为旋转矩阵"""
        angle = torch.norm(axis_angles, dim=1, keepdim=True)
        axis = axis_angles / (angle + 1e-8)
        
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        one_minus_cos = 1 - cos_a
        
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        
        # 旋转矩阵公式
        rot_mat = torch.zeros(
            axis_angles.shape[0], 3, 3, dtype=axis_angles.dtype, device=axis_angles.device
        )
        
        rot_mat[:, 0, 0] = cos_a[:, 0] + one_minus_cos[:, 0] * x * x
        rot_mat[:, 0, 1] = one_minus_cos[:, 0] * x * y - sin_a[:, 0] * z
        rot_mat[:, 0, 2] = one_minus_cos[:, 0] * x * z + sin_a[:, 0] * y
        
        rot_mat[:, 1, 0] = one_minus_cos[:, 0] * y * x + sin_a[:, 0] * z
        rot_mat[:, 1, 1] = cos_a[:, 0] + one_minus_cos[:, 0] * y * y
        rot_mat[:, 1, 2] = one_minus_cos[:, 0] * y * z - sin_a[:, 0] * x
        
        rot_mat[:, 2, 0] = one_minus_cos[:, 0] * z * x - sin_a[:, 0] * y
        rot_mat[:, 2, 1] = one_minus_cos[:, 0] * z * y + sin_a[:, 0] * x
        rot_mat[:, 2, 2] = cos_a[:, 0] + one_minus_cos[:, 0] * z * z
        
        return rot_mat
    
    def forward_kinematics(self, local_rotations, root_translation):
        """前向运动学计算全局位置和旋转"""
        batch_size = local_rotations.shape[0]
        global_positions = torch.zeros(batch_size, 3)
        global_rotations = torch.zeros(batch_size, 3, 3)
        
        # 根关节
        global_positions[0] = torch.tensor(root_translation)
        global_rotations[0] = local_rotations[0]
        
        # 遍历所有关节
        for i in range(1, batch_size):
            parent_idx = self.parent_indices[i]
            
            if parent_idx == -1:  # 根关节
                global_positions[i] = torch.tensor(root_translation)
                global_rotations[i] = local_rotations[i]
            else:
                # 全局旋转 = 父关节全局旋转 × 局部旋转
                global_rotations[i] = torch.matmul(
                    global_rotations[parent_idx], local_rotations[i]
                )
                
                # 假设每个关节相对于父关节的位置偏移为0（简化模型）
                # 在实际SMPL-X中，这里会有固定的骨骼长度
                global_positions[i] = global_positions[parent_idx]
        
        return global_positions, global_rotations
