import torch
import numpy as np
from tqdm import tqdm
from .sam2_video_predictor import SAM2VideoPredictor

class SAM2VideoPredictorFATE(SAM2VideoPredictor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_all_frames_to_correct_as_cond = True

    @torch.inference_mode()
    def propagate_in_video_fate(
        self,
        inference_state,
        support_masks,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points and masks across frames in the video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        if start_frame_idx is None:
            start_frame_idx = min(output_dict["cond_frame_outputs"])

        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="Propagate in video"):
            # Add new supporting masks
            if support_masks and ((frame_idx != start_frame_idx and reverse) or (not reverse)):
                inference_state = self._add_support_masks(inference_state, support_masks[frame_idx])
                self.propagate_in_video_preflight(inference_state)
                output_dict = inference_state["output_dict"]
                consolidated_frame_inds = inference_state["consolidated_frame_inds"]
                obj_ids = inference_state["obj_ids"]
                batch_size = self._get_obj_num(inference_state)

            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )

            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )

            # Delete supporting tasks
            self._clean_added_new_masks(inference_state, len(support_masks[frame_idx]))

            yield frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=False
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        # allow_new_object = not inference_state["tracking_has_started"]
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _add_support_masks(self, inference_state, support_masks):
        """Add support masks"""
        init_num_frames = inference_state['num_frames']
        inference_state['images'] = self._add_support_images(inference_state['images'], support_masks)
        inference_state['num_frames'] += len(support_masks)
        mask_added_flag = False
        for idx, (k, s) in enumerate(support_masks.items()):
            actual_labels = sorted(np.unique(s["label"]))[1:]
            if actual_labels:
                mask_added_flag = True
                for actual_label in actual_labels:
                    mask = (s["label"] == actual_label).astype(float)
                    _, out_obj_ids, out_mask_logits = self.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=idx+init_num_frames,
                        obj_id=actual_label,
                        mask=mask,
                    )
        return inference_state

    def _clean_added_new_masks(self, inference_state, n):
        """Remove newly added support masks"""
        num_frames = inference_state['num_frames']
        obj_ids = inference_state["obj_ids"]
        for i in range(num_frames-n, num_frames):
            for j in obj_ids:
                self.clear_all_prompts_in_frame(inference_state, i, j)
        inference_state["images"] = inference_state["images"][:-n, :, :, :]
        inference_state['num_frames'] -= n

    def _add_support_images(self, existing_tensor, sup):
        """Add support images to inference state"""
        new_images_tensor = torch.stack([data['image'] for data in sup.values()], dim=0)
        new_images_tensor = new_images_tensor.to(existing_tensor.device)
        updated_tensor = torch.cat((existing_tensor, new_images_tensor), dim=0)
        return updated_tensor


