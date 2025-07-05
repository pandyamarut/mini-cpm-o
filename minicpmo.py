import queue
from typing import List, Literal, Union
import uuid

import librosa
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoModel, AutoTokenizer
import time

INPUT_OUTPUT_AUDIO_SAMPLE_RATE = 24000

class AudioData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    array: np.ndarray
    sample_rate: int

class MiniCPMo:
    def __init__(self, device: Literal["cpu", "cuda"] = "cuda", model_revision: str = "main"):
        super().__init__()
        
        torch.cuda.set_device(0)
        self.model = (
            AutoModel.from_pretrained(
                "openbmb/MiniCPM-o-2_6",
                trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                revision=model_revision,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
            
            
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "openbmb/MiniCPM-o-2_6", trust_remote_code=True, revision=model_revision, use_fast=True
        )
        
        if device == "cuda":
            self.init_tts()
        
        self.model_ = torch.compile(self.model, mode="max-autotune", fullgraph=True, dynamic=True)
        print(type(self.model_))
        print(isinstance(self.model_, torch._dynamo.eval_frame.OptimizedModule))

# Or check the class name
        print(self.model_.__class__.__name__)
        self._generate_audio = True
        print("✅ MiniCPMo initialized")

    def init_tts(self):
        self.model.init_tts()
        self.model.tts.bfloat16()

    def _prefill_audio(
        self,
        audio_arrays: List[np.ndarray],
    ):
        audio_samples = np.concatenate(audio_arrays).astype(np.float32)
        print(f"prefilling audio with {audio_samples.shape} samples")
        chunk_size = INPUT_OUTPUT_AUDIO_SAMPLE_RATE * 1
        for chunk_start in range(0, len(audio_samples), chunk_size):
            chunk = audio_samples[chunk_start : chunk_start + chunk_size]

            msgs = [{"role": "user", "content": [chunk]}]

            self.model_.streaming_prefill(
                session_id=self.session_id,
                msgs=msgs,
                tokenizer=self._tokenizer,
                
            )

    def _prefill(self, data: List[str | AudioData]):
        try:
            all_text = []
            all_audio = []
            for prefill_data in data:
                if isinstance(prefill_data, str):
                    all_text.append(prefill_data)
                elif isinstance(prefill_data, AudioData):
                    resampled_audio = librosa.resample(
                        prefill_data.array, prefill_data.sample_rate, 24000
                    )
                    all_audio.append(resampled_audio)
                else:
                    raise ValueError(f"._prefill(): prefill_data must be a string or AudioData")

                if all_text:
                    text = " ".join(all_text)
                    self.model_.streaming_prefill(
                        session_id=self.session_id,
                        msgs=[{"role": "user", "content": [text]}],
                        tokenizer=self._tokenizer,
                    )

                if all_audio:
                    self._prefill_audio(audio_arrays=all_audio)

        except Exception as e:
            print(f"_prefill() error: {e}")
            raise e

    async def run_inference(self, prefill_data: List[str | AudioData]):
        print("MiniCPMo _run_inference() function called")
        print(f"prefill_data: {prefill_data}")
        self.session_id = str(uuid.uuid4())
        try:
            if prefill_data:
                self._prefill(data=prefill_data)


            response_generator = self.model_.streaming_generate(
                session_id=self.session_id,
                tokenizer=self._tokenizer,
                temperature=0.1,
                generate_audio=self._generate_audio,
            )

            for response in response_generator:
                response_received_time = time.perf_counter() 
                audio = None
                sample_rate = INPUT_OUTPUT_AUDIO_SAMPLE_RATE
                text = None

                # extract audio from response
                if hasattr(response, "audio_wav"):
                    audio_gen_start = time.perf_counter()  # Start timing
                    has_audio = True
                    sample_rate = getattr(response, "sampling_rate", INPUT_OUTPUT_AUDIO_SAMPLE_RATE)
                    audio = response.audio_wav.cpu().detach().numpy()
                    audio_data = AudioData(
                        array=audio,
                        sample_rate=sample_rate,
                    )
                    audio_gen_time = time.perf_counter() - audio_gen_start
                    print(f"Audio generation time: {audio_gen_time:.4f}s")  # Show timing
                    print(f"Audio data: {audio_data}")
                    yield audio_data
                
                # check for text
                if isinstance(response, dict):
                    text = response.get("text")
                elif hasattr(response, "text"):
                    text = response.text

                # put audio in output queue
                # if audio is not None:
                #     audio_data = AudioData(
                #         array=audio,
                #         sample_rate=sample_rate,
                #     )   
                    
                #     yield audio_data

                # put text in output queue
                if isinstance(text, str) and text:
                    has_text = True
                    yield text

            yield None

        except Exception as e:
            print(f"_run_inference() error: {e}")
            yield None
