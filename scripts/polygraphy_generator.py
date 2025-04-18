import os
import tensorrt as trt
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PolygraphyCmdGen")


class TRTNetworkInspector:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)

    def create_network(self, onnx_path):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                log.error(f"Failed to parse ONNX: {onnx_path}")
                for i in range(parser.num_errors):
                    log.error(parser.get_error(i))
                return None, None

        inputs = [(network.get_input(i).name, network.get_input(i).shape)
                  for i in range(network.num_inputs)]
        outputs = [network.get_output(i).name for i in range(network.num_outputs)]
        return inputs, outputs


def shape_str(shape):
    return ",".join(str(dim if dim > 0 else 1) for dim in shape)


def gen_polygraphy_cmd(onnx_path, engine_path, inputs, outputs):
    cmd = [
        "polygraphy convert",
        "-v",
        "--convert-to trt",
        "--model-type onnx",
        "--fp16"
    ]

    # Add dynamic shape info
    for name, shape in inputs:
        min_shape = shape_str([1 if s > 0 else 1 for s in shape])
        opt_shape = shape_str([min(4, s) if s > 0 else 4 for s in shape])
        max_shape = shape_str([max(8, s * 2) if s > 0 else 128 for s in shape])
        cmd.append(f"--trt-min-shapes {name}:[{min_shape}]")
        cmd.append(f"--trt-opt-shapes {name}:[{opt_shape}]")
        cmd.append(f"--trt-max-shapes {name}:[{max_shape}]")

    if outputs:
        cmd.append("--trt-outputs " + " ".join(outputs))

    cmd.append(f"-o \"{engine_path}\"")
    cmd.append(f"\"{onnx_path}\"")
    return " \\\n    ".join(cmd)


def main():
    onnx_dir = "/workspace/FasterLivePortrait/checkpoints/liveportrait_onnx"
    inspector = TRTNetworkInspector()

    for file in os.listdir(onnx_dir):
        if file.endswith(".onnx"):
            onnx_path = os.path.join(onnx_dir, file)
            engine_path = onnx_path.replace(".onnx", ".trt")

            inputs, outputs = inspector.create_network(onnx_path)
            if inputs is None:
                continue

            cmd = gen_polygraphy_cmd(onnx_path, engine_path, inputs, outputs)
            print("\n# === Polygraphy command for", file)
            print(cmd)


if __name__ == "__main__":
    main()
