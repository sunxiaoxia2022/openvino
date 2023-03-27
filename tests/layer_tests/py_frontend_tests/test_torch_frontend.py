# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from openvino.frontend import FrontEndManager
from openvino.runtime import PartialShape, Type


class aten_relu(torch.nn.Module):
    def forward(self, x):
        return x, torch.nn.functional.relu(x)


def get_scripted_model(model):
    with torch.no_grad():
        model = torch.jit.script(model)
        model.eval()
        model = torch.jit.freeze(model)
        print(model.inlined_graph)  # will help debugging
        return model


def test_pytorch_fe_set_input_shape():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    om = fe.convert(im)
    assert om.get_parameters()[0].get_partial_shape(
    ) == PartialShape([1, 2, 3, 4])


def test_pytorch_fe_set_input_type():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_element_type(place, Type.f32)
    om = fe.convert(im)
    assert om.get_parameters()[0].get_element_type() == Type.f32


def test_pytorch_fe_set_input_value():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    im.set_element_type(place, Type.f32)
    im.set_tensor_value(place, np.random.randn(1, 2, 3, 4).astype(np.float32))
    om = fe.convert(im)
    assert len(om.get_parameters()) == 0


def test_pytorch_telemetry():
    from openvino.frontend import TelemetryExtension
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    class MockTelemetry:
        def __init__(self, stat):
            self.stat = stat

        def send_event(self, *arg, **kwargs):
            self.stat["send_event"] += 1

        def send_error(self, *arg, **kwargs):
            self.stat["send_error"] += 1

        def send_stack_trace(self, *arg, **kwargs):
            self.stat["send_stack_trace"] += 1

    def add_ext(front_end, stat):
        tel = MockTelemetry(stat)
        front_end.add_extension(TelemetryExtension("mock",
                                                   tel.send_event,
                                                   tel.send_error,
                                                   tel.send_stack_trace))

    tel_stat = {"send_event": 0, "send_error": 0, "send_stack_trace": 0}
    # Ensure that MockTelemetry object is alive and can receive events (due to callbacks hold the object)
    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    add_ext(fe, tel_stat)
    im = fe.load(decoder)
    fe.convert(im)
    assert tel_stat["send_event"] == 2
    assert tel_stat["send_error"] == 0
    assert tel_stat["send_stack_trace"] == 0
