# train_comisr.py

import tensorflow.compat.v1 as tf
tf.disable_eager_execution() # COMISR uses TF1 graph mode
import tensorflow_addons as tfa # For dense_image_warp

import numpy as np
import os
import glob
import argparse
import time
from PIL import Image

# Assuming model.py and ops.py are in the same directory or accessible in PYTHONPATH
from comisr.lib import model
from comisr.lib import ops

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='COMISR Training Script')
parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory of the training dataset')
parser.add_argument('--val_dataset_dir', type=str, default=None, help='Root directory of the validation dataset')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_x2', help='Directory to save checkpoints')
parser.add_argument('--log_dir', type=str, default='./logs_x2', help='Directory to save TensorBoard logs')
parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to a pre-trained model to fine-tune (e.g., original x4 model.ckpt)')

parser.add_argument('--vsr_scale', type=int, default=2, help='Super-resolution scale (2 for x2)')
parser.add_argument('--num_resblock', type=int, default=10, help='Number of residual blocks in generator')
parser.add_argument('--sequence_length', type=int, default=5, help='Number of LR frames in a sequence for training')
parser.add_argument('--hr_target_index', type=int, default=2, help='Index of the HR target frame in the sequence (0 to sequence_length-1), typically middle one.')


parser.add_argument('--lr_patch_size_h', type=int, default=64, help='Height of LR patches for training')
parser.add_argument('--lr_patch_size_w', type=int, default=64, help='Width of LR patches for training')

parser.add_argument('--batch_size', type=int, default=8, help='Batch size') # Paper mentions 16, but might need >1 GPU
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate') # Paper uses 5e-5
parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer beta2')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--content_loss_weight', type=float, default=20.0, help='Weight for HR content loss (beta in paper)')
parser.add_argument('--warp_loss_weight', type=float, default=1.0, help='Weight for LR warp loss (gamma in paper)')

parser.add_argument('--save_freq_epochs', type=int, default=1, help='Save checkpoint every N epochs')
parser.add_argument('--log_freq_steps', type=int, default=100, help='Log to TensorBoard every N steps')
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for data loading')

args = parser.parse_args()

# Derived HR patch size
HR_PATCH_SIZE_H = args.lr_patch_size_h * args.vsr_scale
HR_PATCH_SIZE_W = args.lr_patch_size_w * args.vsr_scale

# --- Data Loading and Preprocessing ---
def _parse_image(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32) / 255.0 # Normalize to [0, 1]
    return image

def load_and_preprocess_sequence(lr_image_paths, hr_image_path_target):
    lr_images = []
    for p in tf.unstack(lr_image_paths): # unstack tensor of paths
        lr_images.append(_parse_image(p))
    lr_sequence = tf.stack(lr_images) # [sequence_length, H, W, C]

    hr_target = _parse_image(hr_image_path_target)

    # Random crop (simplified, assumes LR images are already small enough or you add resize)
    # For proper random crop, you'd crop LR and HR consistently.
    # This example assumes LR frames are already patch_size or will be cropped.
    # A more robust cropping:
    combined = tf.concat([lr_sequence, tf.expand_dims(hr_target, 0)], axis=0) # Temp stack for synced crop

    # Calculate crop offsets
    all_frames_h = tf.shape(lr_sequence)[1]
    all_frames_w = tf.shape(lr_sequence)[2]

    # Ensure patch size is not larger than image
    max_lr_h = all_frames_h - args.lr_patch_size_h
    max_lr_w = all_frames_w - args.lr_patch_size_w
    
    offset_h_lr = tf.cond(max_lr_h > 0,
                          lambda: tf.random.uniform([], 0, max_lr_h, dtype=tf.int32),
                          lambda: tf.constant(0, dtype=tf.int32))
    offset_w_lr = tf.cond(max_lr_w > 0,
                          lambda: tf.random.uniform([], 0, max_lr_w, dtype=tf.int32),
                          lambda: tf.constant(0, dtype=tf.int32))


    offset_h_hr = offset_h_lr * args.vsr_scale
    offset_w_hr = offset_w_lr * args.vsr_scale

    lr_sequence_cropped = lr_sequence[:,
                                      offset_h_lr : offset_h_lr + args.lr_patch_size_h,
                                      offset_w_lr : offset_w_lr + args.lr_patch_size_w,
                                      :]
    hr_target_cropped = hr_target[offset_h_hr : offset_h_hr + HR_PATCH_SIZE_H,
                                  offset_w_hr : offset_w_hr + HR_PATCH_SIZE_W,
                                  :]
    
    lr_sequence_cropped.set_shape([args.sequence_length, args.lr_patch_size_h, args.lr_patch_size_w, 3])
    hr_target_cropped.set_shape([HR_PATCH_SIZE_H, HR_PATCH_SIZE_W, 3])

    # Optional: Random horizontal flip
    do_flip = tf.random.uniform([]) > 0.5
    lr_sequence_cropped = tf.cond(do_flip, lambda: tf.image.flip_left_right(lr_sequence_cropped), lambda: lr_sequence_cropped)
    hr_target_cropped = tf.cond(do_flip, lambda: tf.image.flip_left_right(hr_target_cropped), lambda: hr_target_cropped)

    return lr_sequence_cropped, hr_target_cropped


def create_dataset(dataset_dir, is_training=True):
    scene_dirs = sorted(glob.glob(os.path.join(dataset_dir, "scene*")))
    if not scene_dirs:
        raise ValueError(f"No 'scene*' directories found in {dataset_dir}")

    all_lr_sequences = []
    all_hr_targets = []

    for scene_dir in scene_dirs:
        lr_frames_path = os.path.join(scene_dir, "lr")
        hr_frames_path = os.path.join(scene_dir, "hr")
        
        lr_frame_files = sorted(glob.glob(os.path.join(lr_frames_path, "*.png")))
        hr_frame_files = sorted(glob.glob(os.path.join(hr_frames_path, "*.png")))

        if not lr_frame_files or not hr_frame_files:
            print(f"Warning: No frames found in {scene_dir}, skipping.")
            continue
        
        num_frames = min(len(lr_frame_files), len(hr_frame_files))

        for i in range(num_frames - args.sequence_length + 1):
            lr_seq_paths = lr_frame_files[i : i + args.sequence_length]
            hr_target_path = hr_frame_files[i + args.hr_target_index] # Ensure this index makes sense

            all_lr_sequences.append(lr_seq_paths)
            all_hr_targets.append(hr_target_path)
    
    if not all_lr_sequences:
        raise ValueError(f"No training sequences could be created from {dataset_dir}")

    dataset = tf.data.Dataset.from_tensor_slices((all_lr_sequences, all_hr_targets))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(all_lr_sequences)) # Shuffle files

    dataset = dataset.map(load_and_preprocess_sequence, num_parallel_calls=args.num_threads)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

# --- Model and Loss Definition ---
def build_train_graph(lr_sequence_ph, hr_target_ph):
    batch_size_tf = tf.shape(lr_sequence_ph)[0]
    
    # Initial states (for t=0)
    # Paper: "pre_gen" (previous HR output) and "pre_inputs" (previous LR input)
    # For t=0, pre_inputs can be the first LR frame, pre_gen can be its bicubic upscaled version
    initial_lr_frame = lr_sequence_ph[:, 0, :, :, :]
    
    # Initialize previous HR output (h_state) with bicubic of the first LR frame in the sequence
    # This needs to be batch-aware. ops.bicubic_x expects [B,H,W,C]
    h_state = ops.bicubic_x(initial_lr_frame, scale=args.vsr_scale) # [B, H_hr, W_hr, C]
    h_state = ops.preprocess(h_state) # [-1, 1]

    lr_prev_input = ops.preprocess(initial_lr_frame) # [-1, 1]

    total_content_loss = 0.0
    total_warp_loss = 0.0
    final_hr_prediction = None # Will store the HR prediction for the target frame

    for t in range(args.sequence_length):
        lr_curr_orig = lr_sequence_ph[:, t, :, :, :] # Current LR frame [0,1]
        lr_curr = ops.preprocess(lr_curr_orig)      # Current LR frame [-1,1]

        # 1. Flow Estimation (FNet)
        # fnet_input expects concatenation of current and previous LR frames
        fnet_input = tf.concat([lr_prev_input, lr_curr], axis=-1)
        with tf.variable_scope("fnet", reuse=tf.AUTO_REUSE): # Ensure fnet weights are reused
            # fnet output is LR flow, scaled (e.g. by 24.0 in model.py for TecoGAN)
            # For warping, this flow needs to be scaled by vsr_scale
            lr_flow_raw = model.fnet(fnet_input) # Output range e.g. [-24, 24]

        # 2. Warp previous HR output (h_state) using HR flow
        # Upscale LR flow to HR flow. Paper uses deconvolution.
        # Simpler: bicubic upscale for flow.
        # The scale of flow vectors also needs to be adjusted by vsr_scale.
        hr_flow_upscaled = ops.bicubic_x(lr_flow_raw, scale=args.vsr_scale) * float(args.vsr_scale)
        
        warped_hr = tfa.image.dense_image_warp(h_state, hr_flow_upscaled)

        # 3. Laplacian Enhancement (from inference code / paper Sec 3.3)
        # The inference code uses ops.extract_detail_ops AFTER dense_image_warp.
        # The paper Eq 4 suggests ÎHR = ÎHR + a(ÎHR – G(ÎHR,σ = 1.5)) on the *intermediate* HR.
        # Let's use the inference code's approach as it's likely the implemented one.
        warped_hr_enhanced = warped_hr + ops.extract_detail_ops(warped_hr, sigma=1.5) # sigma from paper

        # 4. Prepare input for Generator
        # Space-to-depth on warped_hr_enhanced
        s2d_warped_hr = tf.space_to_depth(warped_hr_enhanced, args.vsr_scale)
        
        # Concatenate current LR (original scale, preprocessed) and s2d_warped_hr
        # Generator input: [current_lr_frame, s2d_warped_hr_from_previous_step]
        # Note: COMISR inference code uses (inputs_raw, transpose_pre) where inputs_raw = current LR
        # transpose_pre is space_to_depth(pre_warp) where pre_warp is the enhanced warped previous HR
        # lr_curr is already preprocessed to [-1, 1]
        generator_input = tf.concat([lr_curr, s2d_warped_hr], axis=-1)

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE): # Ensure generator weights are reused
            # generator_f outputs image in [-1, 1] range, adds bicubic of LR
            # For training, we want the raw network output before adding bicubic if we compare to HR ground truth.
            # However, generator_f in model.py *does* add bicubic internally.
            # So, the hr_pred will be the final output.
            hr_pred_raw_net = model.generator_f_encoder(generator_input, args.num_resblock)
            hr_pred = model.generator_f_decoder(hr_pred_raw_net, generator_input, 3, args.vsr_scale)


        # Store prediction if this is the target frame
        if t == args.hr_target_index:
            final_hr_prediction = hr_pred # This is in [-1,1]

        # Update states for next iteration
        h_state = hr_pred # This is the new "pre_gen"
        lr_prev_input = lr_curr # This is the new "pre_inputs"

        # Calculate LR Warp Loss (Paper Eq. 6: L2 between current LR and warped previous LR)
        # Need to warp previous LR frame (lr_sequence_ph[:, t-1,...]) using lr_flow_raw
        if t > 0: # Warp loss makes sense from the second frame onwards
            prev_lr_for_warp = ops.preprocess(lr_sequence_ph[:, t-1, :, :, :])
            warped_lr_prev = tfa.image.dense_image_warp(prev_lr_for_warp, lr_flow_raw) # Use raw LR flow
            
            # Loss between warped_lr_prev and lr_curr (current LR frame)
            # Both should be in [-1,1] space
            warp_loss_t = tf.reduce_mean(tf.square(warped_lr_prev - lr_curr))
            total_warp_loss += warp_loss_t
            
    # --- Losses ---
    # Content Loss (Paper Eq. 5, simplified for single target HR)
    # hr_target_ph is [0,1], final_hr_prediction is [-1,1]
    hr_target_preprocessed = ops.preprocess(hr_target_ph)
    content_loss = tf.reduce_mean(tf.square(final_hr_prediction - hr_target_preprocessed))
    
    # Average warp loss over sequence (excluding first frame)
    avg_warp_loss = total_warp_loss / float(args.sequence_length -1) if args.sequence_length > 1 else 0.0

    total_loss = (args.content_loss_weight * content_loss +
                  args.warp_loss_weight * avg_warp_loss)
    
    # Deprocess final_hr_prediction for PSNR calculation (optional, for logging)
    final_hr_prediction_deprocessed = ops.deprocess(final_hr_prediction) # [0,1]
    psnr = tf.image.psnr(hr_target_ph, final_hr_prediction_deprocessed, max_val=1.0)


    return total_loss, content_loss, avg_warp_loss, psnr, final_hr_prediction_deprocessed


# --- Main Training ---
def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --- Dataset ---
    train_dataset = create_dataset(args.dataset_dir, is_training=True)
    iterator = tf.data.make_initializable_iterator(train_dataset)
    lr_sequence_batch, hr_target_batch = iterator.get_next()

    # --- Build Graph ---
    total_loss_op, content_loss_op, warp_loss_op, psnr_op, final_pred_op = \
        build_train_graph(lr_sequence_batch, hr_target_batch)

    # --- Optimizer ---
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)
    
    # Get trainable variables (important if fine-tuning and some parts are frozen)
    # For COMISR, typically all fnet and generator vars are trainable.
    train_vars = tf.trainable_variables()
    
    # Compute gradients
    grads_and_vars = optimizer.compute_gradients(total_loss_op, var_list=train_vars)
    # Optional: Clip gradients
    # grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars if grad is not None]
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # --- Summaries for TensorBoard ---
    tf.summary.scalar("total_loss", total_loss_op)
    tf.summary.scalar("content_loss", content_loss_op)
    tf.summary.scalar("warp_loss", warp_loss_op)
    tf.summary.scalar("psnr_train_batch", psnr_op) # PSNR on training batch
    tf.summary.image("LR_input_seq0_frame0", tf.expand_dims(lr_sequence_batch[0,0,:,:,:], 0))
    tf.summary.image("HR_target_frame0", tf.expand_dims(hr_target_batch[0,:,:,:],0))
    tf.summary.image("HR_prediction_frame0", tf.expand_dims(final_pred_op[0,:,:,:],0))
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_dir, tf.get_default_graph())

    # --- Saver ---
    # If fine-tuning from original x4 model, some layers might not match (e.g., generator decoder for x2)
    # Create a dictionary of variables to restore, filtering by name/shape if necessary.
    # For now, assume we either train from scratch or load a compatible x2 checkpoint.
    
    # tf_vars_to_restore = {v.name.split(':')[0]: v for v in tf.global_variables()
    #                       if not v.name.startswith('Adam') and not v.name.startswith('beta1_power') and not v.name.startswith('beta2_power')}
    # # To load EMA variables from official checkpoint:
    # # restore_vars_dict = {}
    # # for v in tf.global_variables():
    # #     if 'ExponentialMovingAverage' in v.name: continue # skip creating them if not using EMA for training
    # #     ema_name = f"{v.name.split(':')[0]}/ExponentialMovingAverage"
    # #     # check if ema_name exists in checkpoint
    # #     # This part is tricky without knowing exact names in official checkpoint.
    # #     # For now, a simple saver.
    saver = tf.train.Saver(max_to_keep=5)


    # --- Session ---
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # For iterator
        
        if args.load_checkpoint_path:
            print(f"Loading checkpoint from {args.load_checkpoint_path}...")
            # Need to inspect the checkpoint to see variable names if loading official x4
            # For x2 training, it's better to train from scratch or a x2 checkpoint
            try:
                # Create a custom saver if variable names in checkpoint differ (e.g., EMA names)
                # For now, simple restore. If it fails, you'll need to map names.
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(args.load_checkpoint_path))
                if ckpt and ckpt.model_checkpoint_path:
                    # If loading official checkpoint, might need to map EMA names
                    # vars_in_ckpt = [name for name, _shape in tf.train.list_variables(args.load_checkpoint_path)]
                    # print("Vars in checkpoint:", vars_in_ckpt)
                    # For simplicity, assuming direct load works or it's an x2 checkpoint.
                    loader = tf.train.Saver() # Or a custom saver with var_dict
                    loader.restore(sess, args.load_checkpoint_path) # Use exact .ckpt path
                    print(f"Checkpoint {args.load_checkpoint_path} loaded successfully.")
                else:
                     print(f"No checkpoint found at {args.load_checkpoint_path}.")
            except Exception as e:
                print(f"Could not load checkpoint: {e}. Training from scratch or ensure compatibility.")


        print("Starting training...")
        for epoch in range(args.epochs):
            sess.run(iterator.initializer) # Re-initialize iterator for each epoch
            epoch_start_time = time.time()
            step = 0
            try:
                while True:
                    step_start_time = time.time()
                    
                    _, total_loss_val, content_loss_val, warp_loss_val, psnr_val, current_global_step = \
                        sess.run([train_op, total_loss_op, content_loss_op, warp_loss_op, psnr_op, global_step])
                    
                    step_duration = time.time() - step_start_time

                    if current_global_step % args.log_freq_steps == 0:
                        summary_str = sess.run(merged_summary_op)
                        summary_writer.add_summary(summary_str, current_global_step)
                        print(f"Epoch: {epoch+1}/{args.epochs}, Step: {current_global_step}, "
                              f"Total Loss: {total_loss_val:.4f}, Content: {content_loss_val:.4f}, Warp: {warp_loss_val:.4f}, "
                              f"PSNR: {psnr_val:.2f} dB, Time/step: {step_duration:.2f}s")
                    step +=1

            except tf.errors.OutOfRangeError:
                # End of epoch
                pass
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s.")

            if (epoch + 1) % args.save_freq_epochs == 0:
                save_path = saver.save(sess, os.path.join(args.checkpoint_dir, "model.ckpt"), global_step=current_global_step)
                print(f"Checkpoint saved to {save_path}")
        
        # Final save
        save_path = saver.save(sess, os.path.join(args.checkpoint_dir, "model_final.ckpt"), global_step=sess.run(global_step))
        print(f"Final checkpoint saved to {save_path}")
        summary_writer.close()

if __name__ == '__main__':
    main()