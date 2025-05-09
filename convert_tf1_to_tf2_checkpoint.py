# convert_tf1_to_tf2_checkpoint.py

import tensorflow as tf
import tensorflow.compat.v1 as tf1 # For TF1 compatibility layer
import os
import argparse

# Assuming model.py and ops.py are refactored and accessible
from comisr.lib.model import FNet as KerasFNet
from comisr.lib.model import Generator as KerasGenerator
# Ops will be used implicitly by the Keras models

# Disable TF2 eager execution for this TF1 loading part
tf1.disable_eager_execution()

def main():
    parser = argparse.ArgumentParser(description='Convert COMISR TF1 checkpoint to TF2 Keras format.')
    parser.add_argument('--tf1_checkpoint_path', type=str, required=True,
                        help='Path to the TF1 checkpoint (e.g., /path/to/model.ckpt-XXXX)')
    parser.add_argument('--tf2_checkpoint_prefix', type=str, required=True,
                        help='Prefix for the output TF2 checkpoint (e.g., /path/to/tf2_comisr_x4/ckpt)')
    parser.add_argument('--num_resblock', type=int, default=10, help='Number of resblocks in generator.')
    parser.add_argument('--vsr_scale', type=int, default=4, help='VSR scale of the TF1 model.')
    parser.add_argument('--use_ema', type=bool, default=True, help='Whether the TF1 checkpoint used EMA variables.')
    args = parser.parse_args()

    print(f"TF1 Checkpoint Path: {args.tf1_checkpoint_path}")
    print(f"TF2 Checkpoint Prefix: {args.tf2_checkpoint_prefix}")
    print(f"VSR Scale: {args.vsr_scale}, Num Resblocks: {args.num_resblock}, Use EMA: {args.use_ema}")


    # --- Step 1: Define placeholders for inputs (needed to build the graph) ---
    # These shapes are arbitrary but should allow the models to be built.
    # FNet input: [batch, H, W, C_lr_prev + C_lr_curr] = [1, 120, 180, 6] for Vid4 x4
    # Generator input: [batch, H_lr, W_lr, C_lr + C_s2d_hr_prev]
    # e.g., [1, 120, 180, 3 + (3 * 4 * 4)] for x4
    
    # Typical LR dimensions for x4 Vid4 (180x120)
    dummy_lr_h, dummy_lr_w = 120, 180
    # Typical HR dimensions for x4 Vid4 (720x480)
    dummy_hr_h, dummy_hr_w = dummy_lr_h * args.vsr_scale, dummy_lr_w * args.vsr_scale

    fnet_input_ph = tf1.placeholder(tf.float32, [None, dummy_lr_h, dummy_lr_w, 6], name='fnet_input_placeholder')
    
    # For generator, C_s2d_hr_prev is 3 channels * scale * scale
    s2d_channels = 3 * args.vsr_scale * args.vsr_scale
    generator_input_ph = tf1.placeholder(tf.float32, [None, dummy_lr_h, dummy_lr_w, 3 + s2d_channels], name='generator_input_placeholder')

    # --- Step 2: Instantiate Keras models within TF1 variable scopes ---
    # The scopes 'fnet' and 'generator' must match those in the TF1 checkpoint.
    # The Keras models internally use layer names which will be prefixed by these scopes.
    
    # FNet
    # The original `inference_and_eval.py` calls fnet under `with tf.variable_scope('fnet'):`
    with tf1.variable_scope('fnet'):
        # Our KerasFNet uses layer names, so the TF1 scope will act as a prefix.
        # Example: 'fnet/fnet_model/conv_1_enc1/kernel:0'
        # The TF1 checkpoint might have 'fnet/autoencode_unit/encoder_1/conv_1/weights:0'
        # This is where mapping might be needed if names don't align perfectly.
        # For now, we assume the Keras layer names might pick up the outer scope.
        keras_fnet_model = KerasFNet(name='fnet_keras_model_scoped') # Give a Keras model name
        _ = keras_fnet_model(fnet_input_ph) # Build the model

    # Generator
    # The original `inference_and_eval.py` calls generator_f under `with tf.variable_scope('generator'):`
    with tf1.variable_scope('generator'):
        keras_generator_model = KerasGenerator(
            num_resblock=args.num_resblock,
            vsr_scale=args.vsr_scale,
            gen_output_channels=3,
            name='generator_keras_model_scoped'
        )
        _ = keras_generator_model(generator_input_ph) # Build the model

    # --- Step 3: Prepare variable mapping for loading TF1 checkpoint ---
    vars_to_restore = {}
    
    # Collect all global variables created (these are from our Keras models within TF1 scopes)
    # We need to map them to the names in the TF1 checkpoint.
    # This is the most intricate part.
    
    print("\nVariables in the Keras models (under TF1 scopes):")
    # for v in tf1.global_variables():
    #     print(f"  Keras var: {v.name}")

    # List variables in the TF1 checkpoint
    print(f"\nVariables in TF1 checkpoint at {args.tf1_checkpoint_path}:")
    try:
        reader = tf1.train.NewCheckpointReader(args.tf1_checkpoint_path)
        var_to_shape_map_tf1 = reader.get_variable_to_shape_map()
        # for tf1_var_name in sorted(var_to_shape_map_tf1.keys()):
        # print(f"  TF1 ckpt var: {tf1_var_name} {var_to_shape_map_tf1[tf1_var_name]}")
    except Exception as e:
        print(f"Could not read TF1 checkpoint variables: {e}")
        return

    # Heuristic for mapping (this will likely need adjustment)
    # TF1 Slim often uses 'weights' and 'biases'. Keras uses 'kernel' and 'bias'.
    # TF1 BatchNorm uses 'moving_mean', 'moving_variance', 'beta', 'gamma'.
    # Keras BatchNorm uses 'moving_mean', 'moving_variance', 'beta', 'gamma'. (Usually matches)

    # Example:
    # TF1 name: generator/generator_unit/input_stage/conv/weights
    # Keras name might be: generator/generator_keras_model_scoped/input_conv/kernel:0
    # TF1 EMA name: generator/generator_unit/input_stage/conv/weights/ExponentialMovingAverage
    
    # Create a list of all variables in the current graph (our Keras model variables)
    current_graph_vars = tf1.global_variables()
    
    # Try to map based on common patterns. This is highly experimental.
    skipped_keras_vars = []
    found_mappings = 0

    for keras_var in current_graph_vars:
        keras_var_name_no_suffix = keras_var.name.split(':')[0] # Remove :0
        
        # Attempt to find a match in the TF1 checkpoint
        # This needs a sophisticated name transformation logic.
        # For simplicity, let's assume a direct load attempt by tf.train.Saver later.
        # If using EMA, the TF1 names will have '/ExponentialMovingAverage' suffix.
        
        tf1_name_to_load = keras_var_name_no_suffix # Base case: try direct name match
        
        # Simplistic transformation based on original script's EMA loading:
        # Keras variable names might be like: `generator/generator_keras_model_scoped/input_conv/kernel`
        # TF1 EMA names are like: `generator/generator_unit/input_stage/conv/weights/ExponentialMovingAverage`
        # This simple replacement is unlikely to work without more structure.

        # For EMA:
        # The original inference script does:
        # restore_vars_dict[v.name[:-2] + '/ExponentialMovingAverage'] = v
        # This implies the Keras var name (v.name[:-2]) needs to be transformed into the TF1 base name
        # and then /ExponentialMovingAverage is appended.

        # Let's try a direct mapping for non-EMA first. If EMA, we'll append.
        potential_tf1_name = keras_var_name_no_suffix # This is our Keras var name
        
        # Heuristic: Keras uses 'kernel', TF1 often 'weights'. Keras 'bias', TF1 'biases'.
        # This is very brittle.
        tf1_candidate = potential_tf1_name.replace('/kernel', '/weights').replace('/bias', '/biases')
        # Further, Keras models have their own name prefix.
        # e.g., 'generator/generator_keras_model_scoped/' vs 'generator/generator_unit/'
        # This requires careful inspection and string replacement.
        
        # Let's rely on the Saver to do its best or list unmapped vars.
        # For EMA, the TF1 variable name *in the checkpoint* would be the one to look for.
        if args.use_ema:
            # Try to construct the TF1 EMA name from the Keras var name
            # This is highly dependent on how Keras names map to TF1 scopes
            # Example: If Keras var is 'scope/layer/kernel:0', TF1 EMA might be 'scope/layer/weights/ExponentialMovingAverage'
            # This is the hardest part. We will use the TF1 saver and let it try to match based on scope.
            pass # Saver will handle EMA loading if var_list is correct

    # For EMA, the original script creates a dict where keys are TF1 EMA names and values are graph vars.
    if args.use_ema:
        print("Attempting to load EMA variables...")
        ema_restore_dict = {}
        for v_graph in tf1.global_variables(): # These are Keras variables in TF1 graph
            # We need to guess the TF1 EMA name that corresponds to v_graph
            # Example: if v_graph.name is "generator/input_conv/kernel:0"
            # TF1 EMA name might be "generator/generator_unit/input_stage/conv/weights/ExponentialMovingAverage"
            # This requires a robust name mapping function.
            # For now, let's assume a simpler scenario or that the non-EMA names are close enough for the Saver.
            # The original `_get_ema_vars` and loading logic implies the graph already has non-EMA vars
            # and the Saver loads EMA values *into* them using a name map.
            
            # Let's try to create the restore dictionary similar to original inference_and_eval.py
            # This assumes the Keras variable names in the TF1 graph `v_graph.name` are somewhat close
            # to the non-EMA TF1 variable names.
            base_name_in_graph = v_graph.name.split(':')[0]
            
            # Try to transform Keras name to TF1 base name (very heuristic)
            # Example: 'generator/generator_keras_model_scoped/input_conv/kernel'
            # might map to 'generator/generator_unit/input_stage/conv/weights' in TF1.
            # This is where the conversion is most fragile.
            # A better approach would be to print all `v_graph.name` and all TF1 checkpoint names
            # and manually create the mapping logic.
            
            # For now, let's assume a more direct mapping is attempted by the Saver
            # if we provide the graph variables directly.
            # The original script's EMA logic:
            # for v in var_list: # var_list = tf.trainable_variables() + moving_vars
            #   restore_vars_dict[v.name[:-2] + '/ExponentialMovingAverage'] = v
            # This means the *keys* in the saver's dict are the EMA names from the checkpoint,
            # and the *values* are the corresponding variables in the current graph.
            
            # We need to iterate over TF1 checkpoint EMA names and map them to Keras vars.
            # This is complex without knowing the exact naming conventions.
            
            # **Alternative for EMA: Load non-EMA, then if you really need EMA, it's harder.**
            # For now, let's try to load what the Saver can match.
            # If `use_ema` is true, the *names in the checkpoint* will have `/ExponentialMovingAverage`.
            # The Saver needs to map `checkpoint_var_name_ema` to `graph_var_name_non_ema`.
            
            # Simplification: try loading with a saver for all global vars.
            # If specific EMA mapping is needed, it gets very complex here.
            print("EMA loading requires precise name mapping. Attempting general load.")
            # The original script's `restore_vars_dict` for EMA:
            # Keys: TF1 EMA names (e.g., "var/ExponentialMovingAverage")
            # Values: Corresponding tf.Variable objects in the *current graph*
            
            # Create a dict for the saver: {checkpoint_var_name: graph_variable_object}
            ema_load_map = {}
            all_graph_vars_map = {v.name.split(':')[0]: v for v in tf1.global_variables()}

            for tf1_ckpt_var_name_full in var_to_shape_map_tf1.keys():
                if '/ExponentialMovingAverage' in tf1_ckpt_var_name_full:
                    tf1_base_name = tf1_ckpt_var_name_full.replace('/ExponentialMovingAverage', '')
                    # Now, try to find a Keras var that corresponds to tf1_base_name
                    # This is where a robust name_tf1_to_keras_equivalent function is needed.
                    # Heuristic:
                    # tf1_base_name: generator/generator_unit/input_stage/conv/weights
                    # keras_var_name: generator/generator_keras_model_scoped/input_conv/kernel
                    # This mapping is non-trivial.
                    # For now, we'll let the standard saver try its best.
                    # If you list vars from both, you can manually build this map.
                    pass # Fallback to default saver behavior
                elif tf1_ckpt_var_name_full in all_graph_vars_map: # Direct match for non-EMA
                     vars_to_restore[tf1_ckpt_var_name_full] = all_graph_vars_map[tf1_ckpt_var_name_full]


            if not vars_to_restore and args.use_ema: # If no direct non-EMA matches, try to build EMA map
                print("Building EMA restore map (experimental)...")
                # This part is highly dependent on exact naming.
                # Iterate Keras vars, try to form TF1 EMA name, and map.
                for keras_var in tf1.global_variables():
                    k_name = keras_var.name.split(':')[0]
                    # Super heuristic:
                    # 'fnet/fnet_keras_model_scoped/enc_conv1_1/...' -> 'fnet/autoencode_unit/encoder_1/...'
                    # 'generator/generator_keras_model_scoped/input_conv/kernel' -> 'generator/generator_unit/input_stage/conv/weights'
                    
                    # This part needs manual configuration based on inspecting variable names
                    # from `print_variables_from_checkpoint` and your Keras model.
                    # For example:
                    # tf1_ema_name = manual_keras_to_tf1_ema_name(k_name)
                    # if tf1_ema_name in var_to_shape_map_tf1:
                    #     vars_to_restore[tf1_ema_name] = keras_var
                    pass # Placeholder for manual mapping logic

            if not vars_to_restore:
                 print("Warning: EMA variable map is empty or incomplete. Load might fail or be partial.")
                 # Fallback to loading all global variables, saver will try to match by name.
                 vars_to_restore = {v.name.split(':')[0]: v for v in tf1.global_variables()}


        else: # Not using EMA, try to match non-EMA names
            print("Attempting to load non-EMA variables...")
            all_graph_vars_map = {v.name.split(':')[0]: v for v in tf1.global_variables()}
            for tf1_ckpt_var_name in var_to_shape_map_tf1.keys():
                # Try direct match or simple transformations
                if tf1_ckpt_var_name in all_graph_vars_map:
                    vars_to_restore[tf1_ckpt_var_name] = all_graph_vars_map[tf1_ckpt_var_name]
                else:
                    # Try Keras common name changes
                    keras_equiv_name = tf1_ckpt_var_name.replace('/weights', '/kernel').replace('/biases', '/bias')
                    if keras_equiv_name in all_graph_vars_map:
                         vars_to_restore[tf1_ckpt_var_name] = all_graph_vars_map[keras_equiv_name]
            if not vars_to_restore:
                 vars_to_restore = {v.name.split(':')[0]: v for v in tf1.global_variables()}


    # --- Step 4: Load weights using tf1.train.Saver ---
    # If vars_to_restore is empty or not well-mapped, this might only load a few variables.
    # If vars_to_restore is populated, it tries to load only those.
    # If you want the saver to try and match everything it can by name:
    if not vars_to_restore:
        print("vars_to_restore map is empty. Saver will attempt to match all graph variables by name.")
        # This relies on Keras layer names (within TF1 scopes) matching TF1 checkpoint var names.
        saver = tf1.train.Saver() # Saves all global variables
    else:
        print(f"Using explicit variable map for restore with {len(vars_to_restore)} entries.")
        saver = tf1.train.Saver(vars_to_restore)


    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer()) # Initialize all vars first
        try:
            saver.restore(sess, args.tf1_checkpoint_path)
            print("TF1 checkpoint restored successfully into Keras models (within TF1 graph).")
        except Exception as e:
            print(f"Error during TF1 checkpoint restoration: {e}")
            print("Proceeding to save TF2 checkpoint with potentially uninitialized/partially initialized weights.")
            print("You MUST inspect the TF1 checkpoint and Keras model variable names to create a correct mapping if this fails.")
            # return # Optionally exit if restore fails critically

        # --- Step 5: Save as TF2 Checkpoint using tf.train.Checkpoint ---
        # Now that weights are loaded into Keras model variables (in TF1 graph),
        # we can save them using the TF2 checkpoint mechanism.
        # We need to switch to Eager mode temporarily or do this part carefully.
        # For simplicity, we'll create tf.train.Checkpoint objects targeting the Keras models.
        
        # Create TF2 Checkpoint object
        # This object needs to track the Keras models themselves.
        tf2_ckpt = tf.train.Checkpoint(fnet=keras_fnet_model, generator=keras_generator_model)
        
        # The save path for tf.train.Checkpoint should be a prefix, not a full file name.
        tf2_save_path = tf2_ckpt.save(file_prefix=args.tf2_checkpoint_prefix)
        print(f"Keras model weights saved in TF2 checkpoint format to prefix: {args.tf2_checkpoint_prefix}")
        print(f"Actual TF2 checkpoint files saved at: {tf2_save_path}")

    print("\nConversion process finished.")
    print("Please verify the TF2 checkpoint by trying to load it into your TF2 inference script.")
    print("If loading fails or gives poor results, the variable name mapping during TF1 load was likely incomplete.")
    print("Inspect TF1 checkpoint variables using `tf.train.list_variables(tf1_ckpt_path)`")
    print("Inspect Keras model variables using `model.summary()` or `[v.name for v in model.variables]` (after building).")
    
    # tf1_ckpt_path = "/path/to/your/original/model.ckpt-XXXX"
    # var_list = tf.train.list_variables(tf1_ckpt_path)
    # for name, shape in var_list:
    #     print(f"TF1 Var: {name}, Shape: {shape}")

if __name__ == '__main__':
    # Helper to print variables from a checkpoint
    # python convert_tf1_to_tf2_checkpoint.py --tf1_checkpoint_path /path/model.ckpt-XXX --print_vars_only
    # This is just an example; the main script doesn't have a --print_vars_only flag directly
    main()