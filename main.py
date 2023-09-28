from train import train
from infer import infer

if __name__ == "__main__":
    from configs.config import *
    
    checkpoint_paths= CHECKPOINTS_DIR/f"{MODEL_NAME}_{IMAGE_SIZE}.ckpt"    
    train(
        model_name= MODEL_NAME,    
        image_size= IMAGE_SIZE,
        batch_size= BATCH_SIZE,
    )
    
    infer(checkpoint_path= checkpoint_paths,
          image_size=IMAGE_SIZE,
          batch_size=BATCH_SIZE)
    