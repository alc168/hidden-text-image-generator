"""Unit tests for ``generate_hidden_text.main``.

``main`` orchestrates the heavy Stable Diffusion / ControlNet pipeline. The
tests replace those collaborators with mocks and verify the orchestration:
device selection, wiring of the pipeline call, and that every expected output
file is written.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image


@pytest.fixture
def wired_module(generate_module, monkeypatch, tmp_path):
    """Patch the heavy collaborators and run inside a temp working directory."""
    monkeypatch.chdir(tmp_path)

    controlnet_model = MagicMock(name="ControlNetModel")
    monkeypatch.setattr(generate_module, "ControlNetModel", controlnet_model)

    pipe = MagicMock(name="pipe")
    # ``pipe = pipe.to(device)`` should keep returning the same mock.
    pipe.to.return_value = pipe
    output = MagicMock(name="output")
    output_image = MagicMock(name="output_image")
    output_image.resize.return_value = MagicMock(name="small_image")
    output.images = [output_image]
    pipe.return_value = output

    pipeline_cls = MagicMock(name="StableDiffusionControlNetPipeline")
    pipeline_cls.from_pretrained.return_value = pipe
    monkeypatch.setattr(
        generate_module, "StableDiffusionControlNetPipeline", pipeline_cls
    )

    canny_detector = MagicMock(name="CannyDetector")
    canny_instance = MagicMock(name="canny_instance")
    canny_instance.return_value = MagicMock(name="control_image")
    canny_detector.return_value = canny_instance
    monkeypatch.setattr(generate_module, "CannyDetector", canny_detector)

    torch_mock = MagicMock(name="torch")
    torch_mock.cuda.is_available.return_value = False
    torch_mock.float16 = "float16"
    torch_mock.float32 = "float32"
    monkeypatch.setattr(generate_module, "torch", torch_mock)

    return {
        "module": generate_module,
        "pipe": pipe,
        "pipeline_cls": pipeline_cls,
        "controlnet_model": controlnet_model,
        "canny_instance": canny_instance,
        "output_image": output_image,
        "torch": torch_mock,
        "tmp_path": tmp_path,
    }


def test_main_runs_pipeline_once(wired_module):
    wired_module["module"].main()
    wired_module["pipe"].assert_called_once()


def test_main_uses_cpu_when_no_cuda(wired_module):
    wired_module["module"].main()
    wired_module["pipe"].to.assert_called_once_with("cpu")


def test_main_uses_cuda_when_available(wired_module):
    wired_module["torch"].cuda.is_available.return_value = True
    wired_module["module"].main()
    wired_module["pipe"].to.assert_called_once_with("cuda")


def test_main_enables_attention_slicing_on_cpu(wired_module):
    wired_module["module"].main()
    wired_module["pipe"].enable_attention_slicing.assert_called_once()


def test_main_skips_attention_slicing_on_gpu(wired_module):
    wired_module["torch"].cuda.is_available.return_value = True
    wired_module["module"].main()
    wired_module["pipe"].enable_attention_slicing.assert_not_called()


def test_main_passes_expected_generation_parameters(wired_module):
    wired_module["module"].main()
    _, kwargs = wired_module["pipe"].call_args
    assert kwargs["controlnet_conditioning_scale"] == 0.8
    assert kwargs["guidance_scale"] == 7.5
    assert kwargs["num_inference_steps"] == 40
    assert kwargs["height"] == 512
    assert kwargs["width"] == 512
    assert kwargs["image"] is wired_module["canny_instance"].return_value


def test_main_writes_all_output_files(wired_module):
    module = wired_module["module"]
    module.main()

    # text_mask.png is a real PIL image saved to disk.
    assert (wired_module["tmp_path"] / "text_mask.png").exists()

    # control_image and the generated images are mocks; assert save() calls.
    wired_module["canny_instance"].return_value.save.assert_called_once_with(
        "control_image.png"
    )
    wired_module["output_image"].save.assert_called_once_with(
        "jeremy_hidden_landscape.png"
    )
    small_image = wired_module["output_image"].resize.return_value
    small_image.save.assert_called_once_with("jeremy_hidden_landscape_small.png")


def test_main_creates_small_preview_at_128(wired_module):
    module = wired_module["module"]
    module.main()
    args, _ = wired_module["output_image"].resize.call_args
    assert args[0] == (128, 128)
    assert args[1] == Image.Resampling.LANCZOS


def test_main_loads_expected_pretrained_models(wired_module):
    module = wired_module["module"]
    module.main()
    controlnet_args, _ = wired_module["module"].ControlNetModel.from_pretrained.call_args
    assert controlnet_args[0] == "lllyasviel/sd-controlnet-canny"
    pipeline_args, pipeline_kwargs = wired_module["pipeline_cls"].from_pretrained.call_args
    assert pipeline_args[0] == "runwayml/stable-diffusion-v1-5"
    assert pipeline_kwargs["controlnet"] is wired_module["controlnet_model"].from_pretrained.return_value
