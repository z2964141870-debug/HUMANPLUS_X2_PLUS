"""Contact-label-free reward terms for an X2 privileged physical generator."""
from __future__ import annotations
from typing import Mapping
import torch

def realized_physics_reward(*,joint_error:torch.Tensor,body_error:torch.Tensor,root_error:torch.Tensor,
                            sole_normal_force:torch.Tensor,sole_horizontal_speed:torch.Tensor,
                            sole_signed_distance:torch.Tensor,robot_weight_n:float)->Mapping[str,torch.Tensor]:
 """Return tracking and physical-validity terms using realized simulator data only.

 There is intentionally no desired/reference contact argument.  Contact timing
 is an output of the physical generator, not copied from Bronze labels.
 """
 if sole_normal_force.shape[-1]!=2 or sole_horizontal_speed.shape[-1]!=2 or sole_signed_distance.shape[-1]!=2:raise ValueError('sole tensors must be left/right pairs')
 support=torch.clamp(sole_normal_force.sum(-1)/float(robot_weight_n),0.,1.)
 load=sole_normal_force/torch.clamp(sole_normal_force.sum(-1,keepdim=True),min=1e-6)
 slip=(load*sole_horizontal_speed.square()).sum(-1)
 penetration=torch.relu(-sole_signed_distance).square().sum(-1)
 return {'joint_tracking':torch.exp(-2.*joint_error.square().mean(-1)),'body_tracking':torch.exp(-10.*body_error.square().mean(-1)),
         'root_tracking':torch.exp(-10.*root_error.square().mean(-1)),'support':support,'loaded_slip_penalty':slip,'penetration_penalty':penetration}
