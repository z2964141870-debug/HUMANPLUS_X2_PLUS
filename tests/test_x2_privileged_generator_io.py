import numpy as np,torch
from src.x2_privileged_generator_reward import realized_physics_reward
from src.x2_privileged_generator_exporter import PhysicalGeneratorRolloutBuffer

def test_reward_uses_realized_contact_and_is_finite():
 r=realized_physics_reward(joint_error=torch.zeros(3,29),body_error=torch.zeros(3,5,3),root_error=torch.zeros(3,6),sole_normal_force=torch.tensor([[300.,300.],[600.,0.],[0.,0.]]),sole_horizontal_speed=torch.zeros(3,2),sole_signed_distance=torch.zeros(3,2),robot_weight_n=600.)
 assert set(r)=={'joint_tracking','body_tracking','root_tracking','support','loaded_slip_penalty','penetration_penalty'} and all(torch.isfinite(v).all() for v in r.values());assert torch.equal(r['support'],torch.tensor([1.,1.,0.]))

def test_exporter_shapes_and_contact_semantics(tmp_path):
 b=PhysicalGeneratorRolloutBuffer();b.append(time_s=0.,qpos=np.zeros(38),qvel=np.zeros(37),pd_target31=np.zeros(31),contact_lr=np.ones(2,bool),normal_force_lr=np.ones(2));b.append(time_s=.02,qpos=np.zeros(38),qvel=np.zeros(37),pd_target31=np.zeros(31),contact_lr=np.array([1,0]),normal_force_lr=np.array([2.,0.]));m=b.save(tmp_path/'a.npz',tmp_path/'a.json',source_hashes={'seed':'x'});assert m['frames']==2 and 'no reference contact labels' in m['contact_semantics']
