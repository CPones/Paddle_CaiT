import paddle
from config import get_config
from cait import build_cait as build_model
import argparse
import warnings

warnings.filterwarnings("ignore", category=Warning)

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--config_dir',
        dest='config_dir',
        help='The directory for configs',
        type=str,
        default='CaiT/configs/cait_xxs24_224.yaml')

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default='CaiT/cait_xxs24_224.pdparams')
    
    parser.add_argument('--pretrained', default=".")  #
    parser.add_argument('--save-inference-dir', default=".")  #

    return parser.parse_args()

def main(args):
    # config files in ./configs/
    config = get_config(args.config_dir)
    # build model
    model = build_model(config)
    model.eval()
    # load pretrained weights
    model_state_dict = paddle.load(args.model_path)
    model.set_state_dict(model_state_dict)
    # dynamic to static
    input_spec = paddle.static.InputSpec([None, 3, 224, 224], 'float32', 'image')
    model = paddle.jit.to_static(model, input_spec=[input_spec])
    # save inference model
    paddle.jit.save(model, 'CaiT/infer/best_model')
    print('Model is saved in infer/best_model.')


if __name__ == '__main__':
    args = parse_args()
    main(args)