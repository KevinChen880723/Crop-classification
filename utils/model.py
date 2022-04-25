from vit_pytorch.cvt import CvT

def create_model_CvT(cfg):
    model = CvT(
        num_classes = cfg['model']['num_classes'],
        s1_emb_dim = cfg['model']['spec']['DIM_EMBED'][0],        # stage 1 - dimension
        s1_emb_kernel = cfg['model']['spec']['PATCH_SIZE'][0],      # stage 1 - conv kernel
        s1_emb_stride = cfg['model']['spec']['PATCH_STRIDE'][0],      # stage 1 - conv stride
        s1_proj_kernel = cfg['model']['spec']['KERNEL_QKV'][0],     # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride = cfg['model']['spec']['STRIDE_KV'][0],  # stage 1 - attention key / value projection stride
        s1_heads = cfg['model']['spec']['NUM_HEADS'][0],           # stage 1 - heads
        s1_depth = cfg['model']['spec']['DEPTH'][0],           # stage 1 - depth
        s1_mlp_mult = cfg['model']['spec']['MLP_RATIO'][0],        # stage 1 - feedforward expansion factor
        s2_emb_dim = cfg['model']['spec']['DIM_EMBED'][1],       # stage 2 - (same as above)
        s2_emb_kernel = cfg['model']['spec']['PATCH_SIZE'][1],
        s2_emb_stride = cfg['model']['spec']['PATCH_STRIDE'][1],
        s2_proj_kernel = cfg['model']['spec']['KERNEL_QKV'][1],
        s2_kv_proj_stride = cfg['model']['spec']['STRIDE_KV'][1],
        s2_heads = cfg['model']['spec']['NUM_HEADS'][1],
        s2_depth = cfg['model']['spec']['DEPTH'][1],
        s2_mlp_mult = cfg['model']['spec']['MLP_RATIO'][1],
        s3_emb_dim = cfg['model']['spec']['DIM_EMBED'][2],       # stage 3 - (same as above)
        s3_emb_kernel = cfg['model']['spec']['PATCH_SIZE'][2],
        s3_emb_stride = cfg['model']['spec']['PATCH_STRIDE'][2],
        s3_proj_kernel = cfg['model']['spec']['KERNEL_QKV'][2],
        s3_kv_proj_stride = cfg['model']['spec']['STRIDE_KV'][2],
        s3_heads = cfg['model']['spec']['NUM_HEADS'][2],
        s3_depth = cfg['model']['spec']['DEPTH'][2],
        s3_mlp_mult = cfg['model']['spec']['MLP_RATIO'][2],
        dropout = 0.
    )
    return model