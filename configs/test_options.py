from configs.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--ckpt_dir',   type=str,
                    default='./ckpt/model.ckpt', 
                    help='load ckpt path')
        parser.add_argument('--data_name', type=str, default=None)
        parser.add_argument('--data_val', type=str, default='val')

        parser.add_argument('--scale_size',  type=int, default=192)
        parser.add_argument('--visualize',  default=False, type=bool)
        return parser


