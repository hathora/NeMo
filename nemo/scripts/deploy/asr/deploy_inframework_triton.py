import argparse
import logging

from nemo.deploy.deploy_pytriton import DeployPyTriton
from nemo.deploy.asr.asr_deployable import ASRDeploy


LOGGER = logging.getLogger("NeMo")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tmn", "--triton_model_name", type=str, default="triton_model")
    parser.add_argument("-tmv", "--triton_model_version", type=int, default=1)
    parser.add_argument("-trp", "--triton_port", type=int, default=8000)
    parser.add_argument("-tha", "--triton_http_address", type=str, default="0.0.0.0")
    parser.add_argument("-mbs", "--max_batch_size", type=int, default=8)
    parser.add_argument("-hp", "--hf_model_id", type=str, default=None)
    parser.add_argument("-np", "--nemo_checkpoint_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()

    LOGGER.setLevel(logging.INFO)
    LOGGER.info(args)

    deployable = ASRDeploy(hf_model_id=args.hf_model_id, nemo_checkpoint_path=args.nemo_checkpoint_path)

    nm = DeployPyTriton(
        model=deployable,
        triton_model_name=args.triton_model_name,
        triton_model_version=args.triton_model_version,
        max_batch_size=args.max_batch_size,
        http_port=args.triton_port,
        address=args.triton_http_address,
    )
    nm.deploy()
    # Block and keep Triton serving until interrupted
    nm.serve()


if __name__ == "__main__":
    main()


