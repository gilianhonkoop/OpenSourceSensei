# from vllm import LLM, SamplingParams
import os
import re
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
import openai
from openai import OpenAI
from llava.model.builder import load_pretrained_model
from PIL import Image
from tensorboardX import SummaryWriter

from motif import allogger

from .annotator_utils import (
    assign_llava_output,
    concatenate_images,
    get_conv_mode_from_model_name,
    get_default_args,
    log_result_dict,
    prep_query_with_image,
    save_img,
    update_default_args,
    encode_array_to_base64,
    encode_imgfile_to_base64,
    AnnotationIdx
)

GPT_PROMPT = "Here are two images in a simulated environment with a robot infront of a desk. \
Your task is to pick between these images based on how interesting they are. Which image is more interesting \
in terms of the showcased behavior? For context following points would constitute interestingness: \
(1) The robot is currently holding an object in its gripper. \
(2) The robot is pushing an object around or pushing a button or opening the drawer or interacting with entities on the desk. \
(3) Objects on the desk are in an interesting configuration: e.g. a stack. \
Being far away from the desk with the robot arm retracted or just stretching your arm without interactions, is a sign the image is not interesting. \
Answer in maximum one word: 0 for image 1, 1 for image 2, 2 for both images and 3 if you have no clue."


class LlavaVisionLanguageModel:
    def __init__(
        self,
        # system_prompt: str,
        # answer_regex: str,
        # retry_prompt: str,
        model_path: str = "liuhaotian/llava-v1.6-34b",
        # num_gpus: int = 8,
        load_in_8bit: bool = False,
        logdir: Optional[str] = None,
        save_img: bool = False,
        image_concat_orient: str = 'vertical',
        image_concat_buffer: int = 28,
        sampling_params: dict = {},
    ) -> None:
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)

        self.image_concat_orient = image_concat_orient
        self.image_concat_buffer = image_concat_buffer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=self.model_name,
                load_8bit=load_in_8bit,
            )
        )

        self.args = get_default_args()
        update_default_args(self.args, sampling_params)
        self.conv_mode = get_conv_mode_from_model_name(self.model_name)

        if self.args.conv_mode is not None and self.conv_mode != self.args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    self.conv_mode, self.args.conv_mode, self.args.conv_mode
                )
            )
        else:
            self.args.conv_mode = self.conv_mode

        self.main_query_old = "Consider these two images.\
            1. What is shown in the top (upper) image?\
            2. What is shown on the bottom (lower) image?\
            3. The goal is for the robot to explore interesting things and interact with objects and drawers.\
            Is there any difference between the top and bottom image in terms of how interesting it is?\
            Think step by step and answer based on what you see.\
            Remember: If the robot arm is retracted or away from the desk, there is no interaction and it is not interesting. The closer the robot arm is to objects or the drawer, the more interesting the image is."

        self.main_query = "Consider these two images.\n \
        1. What is shown in the top (upper) image? Is the robot gripper very close to objects on the desk or the drawer?\n \
        2. What is shown on the bottom (lower) image? Is the robot gripper close to objects on the desk or the drawer?\n \
        3. The goal is for the robot to explore interesting things and interact with objects and drawers. \
        Is there any difference between the top and bottom image in terms of how interesting it is? \
        Think step by step and answer based on what you see. \
        Remember: If the robot arm is retracted or away from the desk, there is no interaction and it is not interesting. \
        The closer the robot gripper is to objects or the drawer, the more interesting the image is. Hovering above the desk is not more interesting than actively reaching out to a certain object."

        # self.answer_regex = answer_regex
        # self.retry_prompt = retry_prompt
        # self.llm = LLM(
        #     model=model_name,
        #     tensor_parallel_size=num_gpus,
        #     dtype="float16",
        #     max_num_batched_tokens=4096,
        # )
        self.logdir = logdir
        self.save_img = save_img
        self.imgdir = None
        # self.system_prompt = system_prompt
        if self.logdir is not None:
            # Create directory for json dumping
            os.makedirs(self.logdir, exist_ok=True)
            self.imgdir = os.path.join(self.logdir, "images")
            if self.save_img:
                os.makedirs(self.imgdir, exist_ok=True)

        # Now using tensorboard TODO change!
        active_logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        active_logger_dir = os.path.join(active_logger.logdir, "events_writer")
        os.makedirs(active_logger_dir, exist_ok=True)
        self.writer = SummaryWriter(active_logger_dir)

    def prep_images(self, images: List[Any]):
        # TODO check if not PIL image!
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)
        return images_tensor, image_sizes

    def generate(
        self, images: List[Any], #logging_indices: Sequence[int] = None
    ) -> List[int]:
        #assert len(images) == len(logging_indices)

        annotation_outputs = np.zeros(
            len(images),
        )

        for image_i, image in enumerate(images):
            # conv = get_conversation_template(self.model_name)
            if isinstance(image, np.ndarray):
                image = image.squeeze()
                if len(image.shape) > 3:
                    img1 = Image.fromarray(image[0,...])
                    img2 = Image.fromarray(image[1,...])
                    image_processed = concatenate_images(img1, img2, self.image_concat_orient, self.image_concat_buffer)  # vertical/horizontal concatenation with image_concat_bufferpx whitespace
                else:
                    image_processed = Image.fromarray(image)

            prepped_query = prep_query_with_image(self.main_query, self.model)

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], prepped_query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor, image_sizes = self.prep_images([image_processed])

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            initial_output = outputs[0].strip()

            new_message = "Based on the questions below:\n \
            1. What is shown in the top (upper) image?\n \
            2. What is shown on the bottom (lower) image?\n \
            3. The goal is for the robot to explore interesting things and interact with objects and drawers. Is there any difference between the top and bottom image in terms of how interesting it is?\
            You answered: {}\
            Is the top or bottom image more interesting? Reply maximum in 1 word top or bottom. Reply -1 if the text is unsure or there is no difference. ".format(
                initial_output
            )

            # Generate output with the new message:

            conv = conv_templates[self.args.conv_mode].copy()
            conv.append_message(conv.roles[0], prepped_query)
            conv.append_message(conv.roles[1], outputs[0].strip())
            conv.append_message(conv.roles[0], new_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            with torch.inference_mode():
                input_ids = (
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.args.temperature > 0 else False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_new_tokens,
                    use_cache=True,
                )

            final_outputs = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            llava_annotation = assign_llava_output(final_outputs[0].strip())
            annotation_outputs[image_i] = llava_annotation

            result_dict = {
                "pair": str(
                    image_i
                ),  # TODO mAke sure that this is the propper logging index!
                "initial_answer": initial_output,
                "llava_output": final_outputs[0].strip(),
                "llava_distilled_output": str(llava_annotation),
                # "gt_annotation": str(preferences_dataset[pair_i]),
            }

            if self.logdir is not None:
                log_result_dict(self.logdir, result_dict)
                if self.save_img:
                    save_img(self.imgdir, image_processed, image_i)

            self.writer.add_scalar("llava/annotation", llava_annotation, image_i)

        self.writer.close()

        return annotation_outputs
        #     # *****************
        #     conv.system = self.system_prompt
        #     conv.append_message(conv.roles[0], message)
        #     conv.append_message(conv.roles[1], None)
        #     prompt = conv.get_prompt()
        #     prompts.append(prompt)
        #     convs.append(conv)

        # sampling_params = SamplingParams(
        #     top_k=50, max_tokens=4096, temperature=0.1, top_p=0.95, stop=conv.stop_str
        # )
        # outputs = self.llm.generate(prompts, sampling_params)
        # # Parse all the outputs
        # cleaned_outputs = np.full(len(messages), AnnotationIdx.UNKOWN)
        # indexes_to_retry = []
        # prompts_to_retry = []
        # print("Retrying prompts")
        # for i, output in enumerate(outputs):
        #     text_answer = output.outputs[0].text
        #     result = re.search(self.answer_regex, text_answer)
        #     conv = convs[i]
        #     conv.append_message("", text_answer)
        #     if result:
        #         try:
        #             best_sequence = int(result.group(1))
        #             if best_sequence == 1:
        #                 best_sequence = AnnotationIdx.FIRST
        #             elif best_sequence == 2:
        #                 best_sequence = AnnotationIdx.SECOND
        #         except ValueError:
        #             best_sequence = AnnotationIdx.TIE
        #         cleaned_outputs[i] = best_sequence
        #     else:
        #         # Ask the model again
        #         conv.append_message(conv.roles[0], self.retry_prompt)
        #         conv.append_message(conv.roles[1], None)
        #         prompt = conv.get_prompt()
        #         prompts_to_retry.append(prompt)
        #         indexes_to_retry.append(i)
        # # Retry the prompts that were not good
        # second_batch = self.llm.generate(prompts_to_retry, sampling_params)
        # for i, output in zip(indexes_to_retry, second_batch):
        #     text_answer = output.outputs[0].text
        #     convs[i].append_message("", text_answer)
        #     result = re.search(self.answer_regex, text_answer)
        #     if result:
        #         try:
        #             best_sequence = int(result.group(1))
        #             if best_sequence == 1:
        #                 best_sequence = AnnotationIdx.FIRST
        #             elif best_sequence == 2:
        #                 best_sequence = AnnotationIdx.SECOND
        #         except ValueError:
        #             best_sequence = AnnotationIdx.TIE
        #         cleaned_outputs[i] = best_sequence
        # if self.logdir is not None and logging_indices is not None:
        #     # Log the conversations
        #     for conv, idx in zip(convs, logging_indices):
        #         text_conv = conv.get_prompt()
        #         with open(os.path.join(self.logdir, str(idx) + ".txt"), "w") as f:
        #             f.write(text_conv)
        # return cleaned_outputs


class MockLanguageModel:
    def __init__(
        self,
        logdir: Optional[str] = None,
        save_img: bool = False,
    ) -> None:
        self.logdir = logdir
        self.save_img = save_img
        self.imgdir = None
        # self.system_prompt = system_prompt
        if self.logdir is not None:
            # Create directory for json dumping
            os.makedirs(self.logdir, exist_ok=True)
            self.imgdir = os.path.join(self.logdir, "images")
            if self.save_img:
                os.makedirs(self.imgdir, exist_ok=True)

        # Now using tensorboard TODO change!
        active_logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        active_logger_dir = os.path.join(active_logger.logdir, "events_writer")
        os.makedirs(active_logger_dir, exist_ok=True)
        self.writer = SummaryWriter(active_logger_dir)

    def generate(
        self, images: List[Any], #logging_indices: Sequence[int] = None
    ) -> List[int]:
        # return np.random.randint(0, 2, len(images))
        annotation_outputs = np.random.randint(0, 2, len(images))
        for image_i in range(len(images)):
            pref_random = annotation_outputs[image_i]
            self.writer.add_scalar("random/annotation", pref_random, image_i)
        self.writer.close()
        return annotation_outputs



class GPT4LanguageModel:
    def __init__(
        self,
        model: str = "gpt-4-turbo-2024-04-09",
        detail: str = "low",
        temperature: float = 0.2,
        logdir: Optional[str] = None,
        save_img: bool = False,
    ) -> None:

        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.model = model
        self.detail = detail
        self.temperature = temperature

        self.client = OpenAI()

        self.logdir = logdir
        self.save_img = save_img
        self.imgdir = None
        # self.system_prompt = system_prompt
        if self.logdir is not None:
            # Create directory for json dumping
            os.makedirs(self.logdir, exist_ok=True)
            self.imgdir = os.path.join(self.logdir, "images")
            if self.save_img:
                os.makedirs(self.imgdir, exist_ok=True)

        # Now using tensorboard TODO change!
        active_logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        active_logger_dir = os.path.join(active_logger.logdir, "events_writer")
        os.makedirs(active_logger_dir, exist_ok=True)
        self.writer = SummaryWriter(active_logger_dir)

    def generate(self, images: List[Any]) -> List[int]:
        annotation_outputs = np.zeros(len(images),)

        for image_i, image_pair in enumerate(images):

            img1 = image_pair[0, 0, ...]
            img2 = image_pair[1, 0, ...]

            try:
                base64_image1 = encode_array_to_base64(img1, resize=None)
                base64_image2 = encode_array_to_base64(img2, resize=None)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{GPT_PROMPT}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image1}",
                                        "detail": self.detail,
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image2}",
                                        "detail": self.detail,
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=100,
                    temperature=self.temperature,
                )
                response_message = response.choices[0].message.content.strip()

                if "0" in response_message:
                    pref_chatgpt = AnnotationIdx.FIRST # 0
                elif "1" in response_message:
                    pref_chatgpt = AnnotationIdx.SECOND # 1
                elif "2" in response_message:
                    pref_chatgpt = AnnotationIdx.TIE # 2
                else:
                    pref_chatgpt = AnnotationIdx.UNKOWN # 3

                # print("CHAT GPT CONVERSION: ", pref_chatgpt, "Pair id: ", i, "True label: ", preferences_gt[i])
                # accuracy_w_gt += (pref_chatgpt == int(preferences_gt[i])) * 1
            except:
                response_message = "error occured"
                pref_chatgpt = AnnotationIdx.UNKOWN # 3

            annotation_outputs[image_i] = pref_chatgpt

            self.writer.add_scalar("gpt/annotation", pref_chatgpt, image_i)

            result_dict = {
                "pair": str(image_i),
                "gpt_response": response_message,
                "gpt_annotation": str(pref_chatgpt),
                # "preferences_gt": str(preferences_gt[i]),
            }

            if self.logdir is not None:
                log_result_dict(self.logdir, result_dict)
                if self.save_img:
                    # a bit redundant but anyways:
                    img1_pil = Image.fromarray(np.uint8(img1))
                    img2_pil = Image.fromarray(np.uint8(img2))

                    img_vconcat = concatenate_images(
                        img1_pil, img2_pil, "vertical", 28
                    )  # vertical concatenation with 50px whitespace
                    # img_vconcat.save(os.path.join(img_dir, f"img_result_{i}_gt_{preferences_gt[i]}_gpt_{pref_chatgpt}.png"))
                    save_img(self.imgdir, img_vconcat, image_i)

            # chatgpt_outputs.append(pref_chatgpt)

        self.writer.close()
        return annotation_outputs

robodesk_prompts_user = {
    "1": "I will show you an image from a simulated environment. Describe the environment to me and what you deem the agent and then explain what types of behavior / observations would be useful and interesting for the agent to explore in this setting? \n",
    "2": "Now I will show you two images. Based on your answer, choose which one of these images is more interesting in terms of the showcased behavior?\nAnswer in maximum one word: 0 for image 1, 1 for image 2, 2 for both images and 3 if you have no clue.",
}

robodesk_prompts_gpt = "The image depicts a simulated environment featuring a robotic arm interacting with a table that has various objects on it. The objects include buttons of different colors, a red ball, a green cube, and a blue rectangular block. The setting appears to be designed for testing robotic manipulation and interaction with objects.\n\n### Agent\nThe robotic arm is the agent in this environment. It is likely programmed to interact with the objects on the table, possibly for tasks involving sorting, stacking, or pressing buttons.\n\n### Useful and Interesting Behaviors/Observations\n1. **Object Manipulation:**\n   - Grasping and moving objects like the ball, cube, and block.\n   - Testing different grips and force applications to handle objects of various shapes and weights.\n\n2. **Button Interaction:**\n   - Pressing buttons to observe effects, such as changes in the environment or triggering specific actions.\n   - Learning which buttons correspond to which actions or outcomes.\n\n3. **Spatial Awareness:**\n   - Navigating the arm around the table without knocking over objects.\n   - Understanding the spatial relationship between objects for tasks like stacking.\n\n4. **Task Learning:**\n   - Completing specific tasks like sorting objects by color or shape.\n   - Learning sequences of actions to achieve complex goals.\n\n5. **Feedback and Adaptation:**\n   - Using sensory feedback to adjust actions in real-time.\n   - Adapting to changes in the environment, such as objects being moved or removed.\n\n6. **Exploration and Experimentation:**\n   - Experimenting with different strategies for interacting with objects.\n   - Exploring the effects of different actions on the environment.\n\nThese behaviors would help the agent develop skills in manipulation, decision-making, and adaptive learning, which are crucial for more complex real-world applications."

class GPT4LanguageModelwHistory:
    def __init__(
        self,
        model: str = "gpt-4-turbo-2024-04-09",
        detail: str = "low",
        temperature: float = 0.2,
        logdir: Optional[str] = None,
        save_img: bool = False,
    ) -> None:

        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.model = model
        self.detail = detail
        self.temperature = temperature

        self.client = OpenAI()

        self.logdir = logdir
        self.save_img = save_img
        self.imgdir = None
        # self.system_prompt = system_prompt
        if self.logdir is not None:
            # Create directory for json dumping
            os.makedirs(self.logdir, exist_ok=True)
            self.imgdir = os.path.join(self.logdir, "images")
            if self.save_img:
                os.makedirs(self.imgdir, exist_ok=True)

        # Now using tensorboard TODO change!
        active_logger = allogger.get_logger(
            scope=self.__class__.__name__, default_outputs=["tensorboard"]
        )
        active_logger_dir = os.path.join(active_logger.logdir, "events_writer")
        os.makedirs(active_logger_dir, exist_ok=True)
        self.writer = SummaryWriter(active_logger_dir)

    def generate(self, images: List[Any]) -> List[int]:
        annotation_outputs = np.zeros(
            len(images),
        )

        # for image_i, (image_pair, info_pair) in enumerate(zip(images, infos)):
        for image_i, image_pair in enumerate(images):
            # removing info pair since we don't have inventory info in Pokemon but all is in the image!

            img1 = image_pair[0, 0, ...]
            img2 = image_pair[1, 0, ...]

            # info1 = info_pair[0, 0, 0]
            # info2 = info_pair[1, 0, 0]

            user_prompt_1 = robodesk_prompts_user["1"]
            # gpt_response_for_context = get_random_gpt_context()
            gpt_response_for_context = robodesk_prompts_gpt
            user_prompt_2 = robodesk_prompts_user["2"]

            context_image = encode_imgfile_to_base64("robodesk_context_image.png")

            try:
                base64_image1 = encode_array_to_base64(img1, resize=None)
                base64_image2 = encode_array_to_base64(img2, resize=None)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{user_prompt_1}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{context_image}",
                                        "detail": self.detail,
                                    },
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{gpt_response_for_context}",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{user_prompt_2}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image1}",
                                        "detail": self.detail,
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image2}",
                                        "detail": self.detail,
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=100,
                    temperature=self.temperature,
                )
                response_message = response.choices[0].message.content.strip()

                length_to_check = max(50, len(response_message))
                if "0" in response_message[-length_to_check:]:
                    pref_chatgpt = AnnotationIdx.FIRST  # 0
                elif "1" in response_message[-length_to_check:]:
                    pref_chatgpt = AnnotationIdx.SECOND  # 1
                elif "2" in response_message[-length_to_check:]:
                    pref_chatgpt = AnnotationIdx.TIE  # 2
                else:
                    pref_chatgpt = AnnotationIdx.UNKOWN  # 3

                # print("CHAT GPT CONVERSION: ", pref_chatgpt, "Pair id: ", i, "True label: ", preferences_gt[i])
                # accuracy_w_gt += (pref_chatgpt == int(preferences_gt[i])) * 1
            except:
                response_message = "error occured"
                pref_chatgpt = AnnotationIdx.UNKOWN  # 3

            annotation_outputs[image_i] = pref_chatgpt

            self.writer.add_scalar("gpt/annotation", pref_chatgpt, image_i)

            result_dict = {
                "pair": str(image_i),
                "gpt_response": response_message,
                "gpt_annotation": str(pref_chatgpt),
                # "preferences_gt": str(preferences_gt[i]),
            }

            if self.logdir is not None:
                log_result_dict(self.logdir, result_dict)
                if self.save_img:
                    # a bit redundant but anyways:
                    img1_pil = Image.fromarray(np.uint8(img1))
                    img2_pil = Image.fromarray(np.uint8(img2))

                    img_vconcat = concatenate_images(
                        img1_pil, img2_pil, "vertical", 28
                    )  # vertical concatenation with 50px whitespace
                    # img_vconcat.save(os.path.join(img_dir, f"img_result_{i}_gt_{preferences_gt[i]}_gpt_{pref_chatgpt}.png"))
                    save_img(self.imgdir, img_vconcat, image_i)

            # chatgpt_outputs.append(pref_chatgpt)

        self.writer.close()
        return annotation_outputs
