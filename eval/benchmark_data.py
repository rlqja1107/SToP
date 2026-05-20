"""
Dataset loaders for video LLM evaluation benchmarks.

Supported benchmarks:
    NextQA, Video-MME, VideoComp (ActivityNet / YouCook2),
    LongVideoBench, MVBench, MLVU, VCGBench , EventHallusion,
"""

import os
import sys
import math
import copy
import threading

import cv2
import numpy as np
import pandas as pd
import torch
import transformers
import decord
import json

from PIL import Image
from decord import VideoReader, cpu
from pruning.LLaVA_NeXT.llava.conversation import conv_templates
from pruning.LLaVA_NeXT.llava.mm_utils import tokenizer_image_token
# from Qwen2_5_VL.qwen_vl_utils.src.qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


# ---------------------------------------------------------------------------
# EvalDataset
# ---------------------------------------------------------------------------

class EvalDataset(object):
    """
    Iterable dataset that loads samples from a supported benchmark and
    returns model-ready tensors via model-specific ``__getitem*__`` methods.

    Args:
        tokenizer: A HuggingFace tokenizer (or ``None`` for API-based models).
        args: Parsed argument namespace (see ``evaluate.py``).
        model: The loaded model instance (needed for device placement).
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args, model=None):
        self.tokenizer = tokenizer
        self.data_list = []
        self.args = args
        self.model = model
        self.root = args.root_dataset
        # ------------------------------------------------------------------
        # NextQA
        # ------------------------------------------------------------------
        if args.dataset == "NextQA":
            data_path = os.path.join(self.root, "VQA/NextQA")
            eval_data = json.load(open(os.path.join(data_path, "val.json"), "r"))
            for data in eval_data:
                video_path = os.path.join(data_path, "NExTVideo", f"{data['video']}.mp4")
                prompt = (
                    f"Question: {data['question']}\nOptions:\n"
                    f"(A) {data['a0']}\n(B) {data['a1']}\n(C) {data['a2']}\n"
                    f"(D) {data['a3']}\n(E) {data['a4']}\n Only give the option."
                )
                self.data_list.append([video_path, prompt, data["answer"], None, None])

        # ------------------------------------------------------------------
        # Video-MME (Short / Medium / Long)
        # ------------------------------------------------------------------
        elif "Video_MME" in args.dataset:
            data_path = os.path.join(self.root, "VQA/Video_MME")
            eval_data = pd.read_parquet(os.path.join(data_path, "test-00000-of-00001.parquet"))
            medium_cnt = 0
            for _, v in eval_data.iterrows():
                video_path = os.path.join(data_path, "video", f"{v['videoID']}.mp4")
                prompt = (
                    "Select the best answer to the following multiple-choice question based on "
                    "the video. Respond with only the letter (A, B, C, or D) of the correct option. "
                    f"\n{v['question']}\n{v['options'][0]}.\n{v['options'][1]}.\n"
                    f"{v['options'][2]}.\n{v['options'][3]}.\nThe best answer is:"
                )
                duration = v["duration"]
                if "Medium" in args.dataset:
                    if duration.lower() in ["short", "long"] or medium_cnt > 149:
                        continue
                    medium_cnt += 1
                self.data_list.append([video_path, prompt, v["answer"], None, None, duration])


        # ------------------------------------------------------------------
        # LongVideoBench
        # ------------------------------------------------------------------
        elif args.dataset == "LongVideoBench":
            data_dir = os.path.join(self.root, "VQA/LongVideoBench")
            annotation = pd.read_parquet(os.path.join(data_dir, "validation-00000-of-00001.parquet"))
            for _, v in annotation.iterrows():
                video_path = os.path.join(data_dir, "videos", v["video_path"])
                prompt = (
                    "Select the best answer to the following multiple-choice question based on "
                    "the video. Respond with only the letter (A, B, C, or D) of the correct option. "
                    f"\n{v['question']}\n(A) {v['option0'].strip(',')}.\n(B) {v['option1'].strip(',')}.\n"
                    f"(C) {v['option2'].strip(',')}.\n(D) {v['option3'].strip(',')}.\n"
                    f"(E) {v['option4'].strip(',')}.\nThe best answer is:"
                )
                self.data_list.append([video_path, prompt, v["correct_choice"], None, None, None])

        # ------------------------------------------------------------------
        # MVBench
        # ------------------------------------------------------------------
        elif args.dataset == "MVBench":
            data_dir = os.path.join(self.root, "VQA/MVBench/json")
            video_dir = os.path.join(self.root, "VQA/MVBench/video")
            for _, v in mvbench_list.items():
                if v[0] == "fine_grained_pose.json":
                    continue
                data_key = v[0].split(".")[0]
                with open(os.path.join(data_dir, v[0]), "r") as f:
                    json_data = json.load(f)
                for data in json_data:
                    prompt = f"Question: {data['question']}\nOptions:\n"
                    for idx, c in enumerate(data["candidates"]):
                        prompt += f"({chr(ord('A') + idx)}) {c}\n"
                    prompt += "Only give the best option."
                    answer = chr(ord("A") + data["candidates"].index(data["answer"]))
                    start = data["start"] * 1_000_000 if v[3] else None
                    end = data["end"] * 1_000_000 if v[3] else None
                    self.data_list.append([
                        os.path.join(video_dir, v[1], data["video"]),
                        prompt, answer, start, end, data_key,
                    ])

        # ------------------------------------------------------------------
        # MLVU (multiple choice)
        # ------------------------------------------------------------------
        elif args.dataset == "MLVU":
            data_dir = os.path.join(self.root, "VQA/MLVU/MLVU/json")
            video_dir = os.path.join(self.root, "VQA/MLVU/MLVU/video")

            for _, v in mlvu_data_list.items():
                type_path = v[0].split(".")[0]
                with open(os.path.join(data_dir, v[0]), "r") as f:
                    json_data = json.load(f)
                for data in json_data:
                    prompt = f"Question: {data['question']}\nOptions:\n"
                    for idx, c in enumerate(data["candidates"]):
                        prompt += f"({chr(ord('A') + idx)}) {c}\n"
                    prompt += "Only give the best option."
                    answer = chr(ord("A") + data["candidates"].index(data["answer"]))
                    self.data_list.append([
                        os.path.join(video_dir, type_path, data["video"]),
                        prompt, answer, None, None, data["question_type"],
                    ])

        # ------------------------------------------------------------------
        # VCGBench
        # ------------------------------------------------------------------
        elif args.dataset == "VCGBench":
            data_dir = os.path.join(self.root, "fine_grained_task/VCGBench")
            video_formats = [".mp4", ".avi", ".mov", ".mkv"]
            def _find_video(name):
                for fmt in video_formats:
                    p = os.path.join(data_dir, "Test_Videos", f"{name}{fmt}")
                    if os.path.exists(p):
                        return p
                return None

            for split, filename in [
                ("generic", "Generic/test-00000-of-00001.parquet"),
                ("temporal", "Temporal/test-00000-of-00001.parquet"),
                ("consistency", "Consistency/test-00000-of-00001.parquet"),
            ]:
                df = pd.read_parquet(os.path.join(data_dir, filename))
                for _, v in df.iterrows():
                    if split == "consistency":
                        question = v["question_2"] if v["question_1"] == "None" else v["question_1"]
                    else:
                        question = v["question"]
                    video_path = _find_video(v["video_name"])
                    if video_path is None: continue
                    self.data_list.append([video_path, question, v["answer"], None, None, split])


        # ------------------------------------------------------------------
        # EventHallusion
        # ------------------------------------------------------------------
        elif args.dataset == "EventHallusion":
            data_path = os.path.join(self.root, "fine_grained_task/EventHallusion")
            question_paths = {
                "interleave": f"{data_path}/questions/mix_questions.json",
                "entire":     f"{data_path}/questions/entire_questions.json",
                "misleading": f"{data_path}/questions/misleading_questions.json",
            }
            video_path = os.path.join(self.root, "fine_grained_task/EventHallusion/videos")
            answer_prompt = "\nPlease answer yes or no:"
            for split, qpath in question_paths.items():
                with open(qpath, "r") as f:
                    input_datas = json.load(f)
                for video_info in input_datas:
                    vid = video_info["id"].replace("mix", "interleave") if "mix" in video_info["id"] else video_info["id"]
                    video_path_ = os.path.join(video_path, split, f"{vid}.mp4")
                    if split != "misleading":
                        self.data_list.append([
                            video_path_, "Please describe this video in detail.",
                            video_info["event_info"]["caption"], None, None, f"{split}_desc",
                        ])
                    for question in video_info["questions"]:
                        self.data_list.append([
                            video_path_, question["question"] + answer_prompt,
                            question["answer"], None, None, f"{split}_qa",
                        ])

        # ------------------------------------------------------------------
        # VideoComp
        # ------------------------------------------------------------------

        elif args.dataset == "VideoComp":
            data_dir = os.path.join(self.root, "fine_grained_task/VideoComp")
            eval_data = json.load(open(os.path.join(data_dir, "activitynet_comp_val.json"), 'r'))
            for data in eval_data:
                video_path = os.path.join(data_dir, "activitynet_video", f"v_{data['video_id']}.mp4")
                if not os.path.isfile(video_path):
                    video_path = os.path.join(data_dir, "activitynet_video", f"v_{data['video_id']}.mkv")
                prompt = data['question']
                answer = data['answer']
                data_key = data['key']
                self.data_list.append([video_path, prompt, answer, data['query_video/start_time'], data['query_video/end_time'], "ActivityNet"])     

            data_dir = os.path.join(self.root, "fine_grained_task/VideoComp")
            eval_data = json.load(open(os.path.join(data_dir, "youcook2_comp_val.json"), 'r'))
            for data in eval_data:
                video_path = os.path.join(data_dir, "raw_videos/video", f"{data['video_id']}.mp4")
                if not os.path.isfile(video_path):
                    video_path = os.path.join(data_dir, "raw_videos/video", f"{data['video_id']}.mkv")
                if not os.path.isfile(video_path):
                    video_path = os.path.join(data_dir, "raw_videos/video", f"{data['video_id']}.webm")
                prompt = data['question']
                answer = data['answer']
                data_key = data['key']
                self.data_list.append([video_path, prompt, answer, data['query_video/start_time'], data['query_video/end_time'], 'YouCook2'])          


        print(f"Dataset '{args.dataset}': {len(self.data_list)} samples loaded.")



    def __len__(self):
        return len(self.data_list)


    def __getitemQwen2_5VL__(self, video_path, prompt, answer, start_time=None, end_time=None):
        if self.args.no_image:
            messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        else:
            messages = [[{"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt}]}]]

        text = self.image_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, self.args, return_video_kwargs=True, start_time=start_time, end_time=end_time)
        inputs = self.image_processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt", **video_kwargs)
        inputs = inputs.to(self.model.device)
        return inputs, None, None, video_path, prompt, answer

    def __getitemQwen2_5VLPrune__(self, video_path, prompt, answer, start_time=None, end_time=None):
        messages = [
            {"role": "system", "content": self.model.system_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "min_pixels": 16 * 28 * 28, "max_pixels": 1024 * 28 * 28},
                {"type": "text", "text": prompt},
            ]},
        ]
        batched_messages = [messages]
        texts = [self.model.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
        image_inputs, video_inputs, _ = self.image_processor(batched_messages, return_video_kwargs=True)
        inputs = self.model.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda" if self.model.device_map == "auto" else self.model.device)
        return inputs, None, None, video_path, prompt, answer

    def __getitemLLaVA_OV__(self, video_path, prompt, answer, start_time=None, end_time=None):
        video_frames = load_video(video_path, max_frames_num=self.args.n_frame)
        frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        video = [frames]

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
        image_sizes = [frame.size for frame in video_frames]
        return input_ids, video, image_sizes, video_path, prompt, answer


    def __getitemQwen2VL__(self, video_path, prompt, answer, start_time=None, end_time=None):
        # model is a Qwen2VLPruneWrapper
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": video_path,
                 "min_pixels": 4 * 28 * 28, "max_pixels": 256 * 28 * 28,
                 "nframes": self.args.n_frame},
                {"type": "text", "text": prompt},
            ]},
        ]
        texts = [self.model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )]
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True
        )
        inputs = self.model.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs
        )
        inputs = inputs.to(self.model.device)
        return inputs, None, None, video_path, prompt, answer



    def __getitemDyCoke__(self, video_path, prompt, answer, start_time, end_time):
        frames = load_video(video_path, self.args.n_frame)
        frames = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensor = [frames]

        image_tokens = " ".join([DEFAULT_IMAGE_TOKEN])
        question = image_tokens + "\n" + prompt
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        input_ids_list = [tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.model.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.model.device)
        return input_ids, image_tensor, [pad_token_ids, attention_masks], video_path, prompt, answer

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data_list):
            raise StopIteration
        i = self.index
        self.index += 1
        video_path  = self.data_list[i][0]
        prompt      = self.data_list[i][1]
        answer      = self.data_list[i][2]
        start_time  = self.data_list[i][3]
        end_time    = self.data_list[i][4]

        if self.args.dataset in [
            "MVBench", 
            "EventHallusion", "VideoHalluc", "VCGBench", 'Video_MME', "VideoComp"
        ]:
            self.eval_type = self.data_list[i][-1]

        backbone = self.args.backbone
        backbone_lower = backbone.lower()
        if backbone in ("LLaVA_Next_Video", "LLaVA_Next_Video_Qwen") or "holitom_llava" in backbone:
            return self.__getitemLLaVA_NV__(video_path, prompt, answer, start_time, end_time)
        elif "qwen2.5-vl" in backbone_lower or "qwen2-vl" in backbone_lower:
            return self.__getitemQwen2VL__(video_path, prompt, answer, start_time, end_time)
        elif "qwen25vl_prune" in backbone:
            return self.__getitemQwen2_5VLPrune__(video_path, prompt, answer, start_time, end_time)
        elif 'onevision' in backbone_lower or 'llava-video' in backbone_lower:
            # LLaVA-OneVision-Qwen2 and LLaVA-Video-Qwen2 share the same llava_qwen
            # architecture and qwen_1_5 conv template, so reuse the OV handler.
            return self.__getitemLLaVA_OV__(video_path, prompt, answer, start_time, end_time)
        elif "dycoke" in backbone:
            # Dycoke cannot be run on the retention ratio 10, 15% since its lowest retention ratio is 25%
            return self.__getitemDyCoke__(video_path, prompt, answer, start_time, end_time)
        else:
            raise ValueError(f"No __getitem__ implementation for model '{backbone}'")


# ---------------------------------------------------------------------------
# Video loading utilities
# ---------------------------------------------------------------------------

def load_video(video_path, max_frames_num,  start_time=None, end_time=None):
    """Load *max_frames_num* frames from a video file or directory."""
    if os.path.isdir(video_path):
        frames, indices = load_video_from_dir(video_path, max_frames_num, start_time, end_time)
        return np.stack(frames)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    
    end_idx = len(vr)
    start_idx = 0
    if end_time:
        fps = vr.get_avg_fps()
        max_duration = end_idx / fps
        start_time /= 1000000
        video_start_clamped = max(0.0, min(start_time, max_duration))
        start_idx = math.ceil(video_start_clamped * fps)
        
        end_time /= 1000000
        video_end_clamped = max(0.0, min(end_time, max_duration))
        end_frame = math.floor(video_end_clamped * fps)
        end_idx = min(end_frame, end_idx)

    frame_idx = torch.linspace(start_idx, end_idx - 1, max_frames_num).round().long().tolist()
    try:
        spare_frames = vr.get_batch(frame_idx).asnumpy()
    except Exception:
        spare_frames = vr.get_batch(frame_idx).numpy()
    return spare_frames


def load_video_from_dir(video_path, max_frames=128,  start_time=None, end_time=None,):
    """Load frames from a directory of images."""
    if start_time is not None and end_time is not None:
        start_time /= 1_000_000
        end_time /= 1_000_000
        start_time = max(0.0, start_time)
        end_time = max(0.0, end_time)
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    frame_files = sorted(os.listdir(video_path))
    vid_fps = 3
    n = len(frame_files)
    f_start = 0 if start_time is None else max(int(start_time * vid_fps) - 1, 0)
    f_end = n - 1 if end_time is None else min(int(end_time * vid_fps) - 1, n - 1)
    frame_indices = list(range(f_start, f_end + 1))
    sampled = [frame_indices[i] for i in frame_sample(len(frame_indices), mode="uniform", num_frames=max_frames)]
    frames = [
        cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[idx])), cv2.COLOR_BGR2RGB)
        for idx in sampled
    ]
    return frames, sampled


def frame_sample(duration: int, mode: str = "uniform", num_frames: int = None,
                 vid_fps: float = None, fps: float = None) -> np.ndarray:
    if mode == "uniform":
        assert num_frames is not None
        if duration <= num_frames:
            return np.arange(duration, dtype=int)
        return np.linspace(0, duration - 1, num_frames, dtype=int)
    elif mode == "fps":
        assert vid_fps is not None and fps is not None
        segment_len = min(int(vid_fps // fps), duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ValueError(f"Unsupported frame sampling mode: '{mode}'")




# ---------------------------------------------------------------------------
# Benchmark metadata tables
# ---------------------------------------------------------------------------

mvbench_list = {
    "Action Sequence":        ("action_sequence.json",       "star/Charades_v1_480/",           "video", True),
    "Action Prediction":      ("action_prediction.json",     "star/Charades_v1_480/",           "video", True),
    "Action Antonym":         ("action_antonym.json",        "ssv2_video/",                     "video", False),
    "Fine-grained Action":    ("fine_grained_action.json",   "Moments_in_Time_Raw/videos/",     "video", False),
    "Unexpected Action":      ("unexpected_action.json",     "FunQA_test/test/",                "video", False),
    "Object Existence":       ("object_existence.json",      "clevrer/video_validation/",       "video", False),
    "Object Interaction":     ("object_interaction.json",    "star/Charades_v1_480/",           "video", True),
    "Object Shuffle":         ("object_shuffle.json",        "perception/videos/",              "video", False),
    "Moving Direction":       ("moving_direction.json",      "clevrer/video_validation/",       "video", False),
    "Action Localization":    ("action_localization.json",   "sta/sta_video/",                  "video", True),
    "Scene Transition":       ("scene_transition.json",      "scene_qa/video/",                 "video", False),
    "Action Count":           ("action_count.json",          "perception/videos/",              "video", False),
    "Moving Count":           ("moving_count.json",          "clevrer/video_validation/",       "video", False),
    "Moving Attribute":       ("moving_attribute.json",      "clevrer/video_validation/",       "video", False),
    "State Change":           ("state_change.json",          "perception/videos/",              "video", False),
    "Fine-grained Pose":      ("fine_grained_pose.json",     "nturgbd/",                        "video", False),
    "Character Order":        ("character_order.json",       "perception/videos/",              "video", False),
    "Egocentric Navigation":  ("egocentric_navigation.json", "vlnqa/",                          "video", False),
    "Episodic Reasoning":     ("episodic_reasoning.json",    "tvqa/frames_fps3_hq/",            "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}

mlvu_data_list = {
    "count":          ("4_count.json",          "/MLVU_all/video/count",          "video"),
    "ego":            ("3_ego.json",            "/MLVU_all/video/ego",            "video"),
    "needle":         ("2_needle.json",         "/MLVU_all/video/needle",         "video"),
    "order":          ("5_order.json",          "/MLVU_all/video/order",          "video"),
    "plotQA":         ("1_plotQA.json",         "/MLVU_all/video/plotQA",         "video"),
    "anomaly_reco":   ("6_anomaly_reco.json",   "/MLVU_all/video/anomaly_reco",   "video"),
    "topic_reasoning":("7_topic_reasoning.json","/MLVU_all/video/topic_reasoning","video"),
}
