import supervisely as sly
import os
from supervisely.nn.inference import CheckpointInfo
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from supervisely.nn.prediction_dto import PredictionBBox
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from supervisely.nn.inference.inference import Inference
from supervisely.geometry.point import Point
import re
import numpy as np
from fastapi import Request
import base64
from io import BytesIO


class Molmo(Inference):
    FRAMEWORK_NAME = "Molmo"
    MODELS = "src/models.json"
    APP_OPTIONS = "src/app_options.yaml"
    INFERENCE_SETTINGS = "src/inference_settings.yaml"

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        checkpoint_path = model_files["checkpoint"]
        if sly.is_development():
            checkpoint_path = "." + checkpoint_path
        self.classes = ["point"]
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_info["meta"]["model_name"],
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=model_source,
        )
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    def extract_points(self, molmo_output, image_w, image_h):
        """Extract points from Molmo output"""
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return all_points

    def predict(self, image_path, settings):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        text_prompt = settings.get("text_prompt", "Count all objects")
        max_new_tokens = settings.get("max_new_tokens", 400)
        inputs = self.processor.process(
            images=[image],
            text=text_prompt,
        )
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"
                ),
                tokenizer=self.processor.tokenizer,
            )
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        points = self.extract_points(generated_text, image.width, image.height)
        return points

    def get_info(self):
        info = super().get_info()
        info["task type"] = "object pointing"
        return info

    def _get_obj_class_shape(self):
        return Point

    def _create_label(self, prediction):
        class_name = "point"
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Point)
            )
            obj_class = self.model_meta.get_obj_class(class_name)
        geometry = sly.Point(row=prediction[1], col=prediction[0])
        label = sly.Label(geometry, obj_class)
        return label

    def serve(self):
        super().serve()
        server = self._app.get_server()

        @server.post("/visual_question_answering")
        def visual_question_answering(request: Request):
            api = request.state.api
            state = request.state.state
            if "image_id" in state:
                image_id = state["image_id"]
                image_np = api.image.download_np(image_id)
                image_pil = Image.fromarray(image_np)
            elif "image_path" in state:
                image_pil = Image.open(state["image_path"])
            elif "image_encoding" in state:
                image_data = base64.b64decode(state["image_encoding"])
                image_pil = Image.open(BytesIO(image_data))
            else:
                raise ValueError(
                    "Request must contain either image_id, image_path or image_encoding!"
                )

            text_prompt = state["text_prompt"]

            inputs = self.processor.process(
                images=[image_pil],
                text=text_prompt,
            )
            inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
            inputs["images"] = inputs["images"].to(torch.bfloat16)
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=400, stop_strings="<|endoftext|>"),
                    tokenizer=self.processor.tokenizer,
                )
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return {"answer": generated_text}

        @server.post("/get_prompt_instructions")
        def get_prompt_instructions(request: Request):
            instructions = (
                "Examples prompts for Molmo image-to-text inference: 'Describe this image.', 'How many apples "
                "can you see on this image?'."
            )
            return {"instructions": instructions}
