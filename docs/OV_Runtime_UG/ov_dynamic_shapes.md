# Dynamic Shapes {#openvino_docs_OV_UG_DynamicShapes}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_NoDynamicShapes


As it was demonstrated in the :doc:`Changing Input Shapes <openvino_docs_OV_UG_ShapeInference>` article, there are models that support changing input shapes before model compilation in ``Core::compile_model``.
Reshaping models provides an ability to customize the model input shape for the exact size required in the end application.
This article explains how the ability of model to reshape can further be leveraged in more dynamic scenarios.

Applying Dynamic Shapes
#######################

Conventional "static" model reshaping works well when it can be done once per many model inference calls with the same shape.
However, this approach does not perform efficiently if the input tensor shape is changed on every inference call. Calling the ``reshape()`` and ``compile_model()`` methods each time a new size comes is extremely time-consuming.
A popular example would be inference of natural language processing models (like BERT) with arbitrarily-sized user input sequences.
In this case, the sequence length cannot be predicted and may change every time inference is called.
Dimensions that can be frequently changed are called *dynamic dimensions*.
Dynamic shapes should be considered, when a real shape of input is not known at the time of the ``compile_model()`` method call.

Below are several examples of dimensions that can be naturally dynamic:

* Sequence length dimension for various sequence processing models, like BERT
* Spatial dimensions in segmentation and style transfer models
* Batch dimension
* Arbitrary number of detections in object detection models output

There are various methods to address input dynamic dimensions through combining multiple pre-reshaped models and input data padding.
The methods are sensitive to model internals, do not always give optimal performance and are cumbersome.
For a short overview of the methods, refer to the :doc:`When Dynamic Shapes API is Not Applicable <openvino_docs_OV_UG_NoDynamicShapes>` page.
Apply those methods only if native dynamic shape API described in the following sections does not work or does not perform as expected.

The decision about using dynamic shapes should be based on proper benchmarking of a real application with real data.
Unlike statically shaped models, dynamically shaped ones require different inference time, depending on input data shape or input tensor content.
Furthermore, using the dynamic shapes can bring more overheads in memory and running time of each inference call depending on hardware plugin and model used.

Handling Dynamic Shapes Natively
################################

This section describes how to handle dynamically shaped models natively with OpenVINO Runtime API version 2022.1 and higher.
There are three main parts in the flow that differ from static shapes:

* Configure the model.
* Prepare data for inference.
* Read resulting data after inference.

Configuring the Model
+++++++++++++++++++++

To avoid the methods mentioned in the previous section, there is a way to specify one or multiple dimensions to be dynamic, directly in the model inputs.
This is achieved with the same reshape method that is used for alternating static shape of inputs.
Dynamic dimensions are specified as ``-1`` or the ``ov::Dimension()`` instead of a positive number used for static dimensions:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:reshape_undefined

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: reshape_undefined

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:reshape_undefined


To simplify the code, the examples assume that the model has a single input and single output.
However, there are no limitations on the number of inputs and outputs to apply dynamic shapes.

Undefined Dimensions "Out Of the Box"
+++++++++++++++++++++++++++++++++++++

Dynamic dimensions may appear in the input model without calling the ``reshape`` method.
Many DL frameworks support undefined dimensions.
If such a model is converted with Model Optimizer or read directly by the ``Core::read_model``, undefined dimensions are preserved.
Such dimensions are automatically treated as dynamic ones.
Therefore, there is no need to call the ``reshape`` method, if undefined dimensions are already configured in the original or the IR model.

If the input model has undefined dimensions that will not change during inference. It is recommended to set them to static values, using the same ``reshape`` method of the model.
From the API perspective, any combination of dynamic and static dimensions can be configured.

Model Optimizer provides identical capability to reshape the model during the conversion, including specifying dynamic dimensions.
Use this capability to save time on calling ``reshape`` method in the end application.
To get information about setting input shapes using Model Optimizer, refer to :doc:`Setting Input Shapes <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`.

Dimension Bounds
++++++++++++++++++++

Apart from a dynamic dimension, the lower and/or upper bounds can also be specified. They define a range of allowed values for the dimension.
The bounds are coded as arguments for the ``ov::Dimension``:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:reshape_bounds

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: reshape_bounds

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:reshape_bounds


Information about bounds gives an opportunity for the inference plugin to apply additional optimizations.
Using dynamic shapes assumes the plugins apply more flexible optimization approach during model compilation.
It may require more time/memory for model compilation and inference.
Therefore, providing any additional information, like bounds, can be beneficial.
For the same reason, it is not recommended to leave dimensions as undefined, without the real need.

When specifying bounds, the lower bound is not as important as the upper one. The upper bound allows inference devices to allocate memory for intermediate tensors more precisely. It also allows using a fewer number of tuned kernels for different sizes.
More precisely, benefits of specifying the lower or upper bound is device dependent.
Depending on the plugin, specifying the upper bounds can be required. For information about dynamic shapes support on different devices, refer to the :ref:`feature support table <devicesupport-feature-support-matrix>`.

If the lower and upper bounds for a dimension are known, it is recommended to specify them, even if a plugin can execute a model without the bounds.

Setting Input Tensors
+++++++++++++++++++++

Preparing a model with the ``reshape`` method is the first step.
The second step is passing a tensor with an appropriate shape to infer request.
This is similar to the :doc:`regular steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>`. However, tensors can now be passed with different shapes for the same executable model and even for the same inference request:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:set_input_tensor

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: set_input_tensor

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:set_input_tensor


In the example above, the ``set_input_tensor`` is used to specify input tensors.
The real dimension of the tensor is always static, because it is a particular tensor and it does not have any dimension variations in contrast to model inputs.

Similar to static shapes, ``get_input_tensor`` can be used instead of ``set_input_tensor``.
In contrast to static input shapes, when using ``get_input_tensor`` for dynamic inputs, the ``set_shape`` method for the returned tensor should be called to define the shape and allocate memory.
Without doing so, the tensor returned by ``get_input_tensor`` is an empty tensor. The shape of the tensor is not initialized and memory is not allocated, because infer request does not have information about the real shape that will be provided.
Setting shape for an input tensor is required when the corresponding input has at least one dynamic dimension, regardless of the bounds.
Contrary to previous example, the following one shows the same sequence of two infer requests, using ``get_input_tensor`` instead of ``set_input_tensor``:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:get_input_tensor

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: get_input_tensor

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:get_input_tensor


Dynamic Shapes in Outputs
+++++++++++++++++++++++++

Examples above are valid approaches when dynamic dimensions in output may be implied by propagation of dynamic dimension from the inputs.
For example, batch dimension in an input shape is usually propagated through the whole model and appears in the output shape.
It also applies to other dimensions, like sequence length for NLP models or spatial dimensions for segmentation models, that are propagated through the entire network.

Whether the output has dynamic dimensions or not can be verified by querying the output partial shape after the model is read or reshaped.
The same applies to inputs. For example:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:print_dynamic

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: print_dynamic

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:print_dynamic


When there are dynamic dimensions in corresponding inputs or outputs, the ``?`` or ranges like ``1..10`` appear.

It can also be verified in a more programmatic way:


.. tab-set::

    .. tab-item:: C++
        :sync: cpp

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.cpp
            :language: cpp
            :fragment: ov_dynamic_shapes:detect_dynamic

    .. tab-item:: Python
        :sync: py

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.py
           :language: python
           :fragment: detect_dynamic

    .. tab-item:: C
        :sync: c

        .. doxygensnippet:: docs/snippets/ov_dynamic_shapes.c
           :language: cpp
           :fragment: ov_dynamic_shapes:detect_dynamic


If at least one dynamic dimension exists in an output of a model, a shape of the corresponding output tensor will be set as the result of inference call.
Before the first inference, memory for such a tensor is not allocated and has the ``[0]`` shape.
If the ``set_output_tensor`` method is called with a pre-allocated tensor, the inference will call the ``set_shape`` internally, and the initial shape is replaced by the calculated shape.
Therefore, setting a shape for output tensors in this case is useful only when pre-allocating enough memory for output tensor. Normally, the ``set_shape`` method of a ``Tensor`` re-allocates memory only if a new shape requires more storage.

@endsphinxdirective