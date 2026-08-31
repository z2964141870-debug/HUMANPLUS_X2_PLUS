from __future__ import annotations

import mink
from native_preprocess import NativeFramePreprocessor


class LimitsFixedGMR:
    """Mixin for GeneralMotionRetargeting that passes cached limits correctly."""

    def _solve_stage(self, tasks, error_function):
        current_error = error_function()
        dt = self.configuration.model.opt.timestep
        velocity = mink.solve_ik(
            self.configuration,
            tasks,
            dt,
            self.solver,
            self.damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(velocity, dt)
        next_error = error_function()
        iteration = 0
        while current_error - next_error > 0.001 and iteration < self.max_iter:
            current_error = next_error
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
            next_error = error_function()
            iteration += 1

    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)
        if self.use_ik_match_table1:
            self._solve_stage(self.tasks1, self.error1)
        if self.use_ik_match_table2:
            self._solve_stage(self.tasks2, self.error2)
        return self.configuration.data.qpos.copy()


class NativePreprocessedGMR(LimitsFixedGMR):
    def _ensure_native_preprocessor(self):
        if hasattr(self, "_native_preprocessor"):
            return
        body_names = tuple(self.human_body_to_task1.keys())
        if set(body_names) != set(self.human_scale_table):
            raise ValueError("Native fast path requires task-1 bodies to match the scale table")
        if set(body_names) != set(self.human_body_to_task2):
            raise ValueError("Native fast path requires both IK stages to use the same human bodies")
        self._native_preprocessor = NativeFramePreprocessor(
            body_names,
            self.human_scale_table,
            self.pos_offsets1,
            self.rot_offsets1,
            self.human_root_name,
        )

    def update_targets(self, human_data, offset_to_ground=False):
        if offset_to_ground:
            raise NotImplementedError("The native prototype does not implement offset_to_ground=True")
        self._ensure_native_preprocessor()
        # The deployed implementation applies table-1 offsets once, then feeds the
        # resulting frames to both IK stages. Preserve that behavior exactly.
        scaled_human_data = self._native_preprocessor(human_data, self.ground_offset)
        self.scaled_human_data = scaled_human_data

        if self.use_ik_match_table1:
            for body_name, task in self.human_body_to_task1.items():
                position, quaternion = scaled_human_data[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position)
                )

        if self.use_ik_match_table2:
            for body_name, task in self.human_body_to_task2.items():
                position, quaternion = scaled_human_data[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position)
                )
