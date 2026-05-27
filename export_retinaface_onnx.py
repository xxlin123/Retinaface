import argparse
import os
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet


def get_output(internals, name):
    output_name = name + "_output"
    outputs = internals.list_outputs()
    if output_name not in outputs:
        print("Cannot find:", output_name)
        print("Available related outputs:")
        for o in outputs:
            if "face_rpn" in o:
                print("  ", o)
        raise KeyError(output_name)
    return internals[output_name]


def build_export_symbol_without_softmax(prefix, epoch, strides=(32, 16, 8), face_landmark=True):
    """
    Load your saved RetinaFace symbol and params.

    Then rebuild an ONNX-friendly inference symbol:

        cls_score_strideX
        bbox_pred_strideX
        landmark_pred_strideX

    Important:
        No SoftmaxActivation in the exported graph.
        Softmax will be done in Python postprocess.
    """

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    internals = sym.get_internals()

    outs = []

    for stride in strides:
        cls_score = get_output(internals, f"face_rpn_cls_score_stride{stride}")
        bbox_pred = get_output(internals, f"face_rpn_bbox_pred_stride{stride}")

        outs.append(cls_score)
        outs.append(bbox_pred)

        if face_landmark:
            landmark_pred = get_output(internals, f"face_rpn_landmark_pred_stride{stride}")
            outs.append(landmark_pred)

    export_sym = mx.sym.Group(outs)

    return export_sym, arg_params, aux_params


def export_retinaface_to_onnx(prefix, epoch, onnx_path, input_size=640):
    export_sym, arg_params, aux_params = build_export_symbol_without_softmax(
        prefix=prefix,
        epoch=epoch,
        strides=(32, 16, 8),
        face_landmark=True,
    )

    tmp_prefix = prefix + "_onnx_no_softmax_tmp"
    mx.model.save_checkpoint(tmp_prefix, 0, export_sym, arg_params, aux_params)

    sym_path = tmp_prefix + "-symbol.json"
    params_path = tmp_prefix + "-0000.params"

    input_shape = [(1, 3, input_size, input_size)]

    print("Exporting ONNX without SoftmaxActivation...")
    print("sym_path:", sym_path)
    print("params_path:", params_path)
    print("onnx_path:", onnx_path)

    converted_model_path = onnx_mxnet.export_model(
        sym=sym_path,
        params=params_path,
        input_shape=input_shape,
        input_type=np.float32,
        onnx_file_path=onnx_path,
    )

    print("Exported ONNX model:", converted_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="./model/retina")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--onnx", type=str, default="./model/retina_no_softmax.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    export_retinaface_to_onnx(
        prefix=args.prefix,
        epoch=args.epoch,
        onnx_path=args.onnx,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()