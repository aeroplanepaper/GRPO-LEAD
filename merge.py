import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def merge_sharded_state_dict(checkpoint_dir, num_shards=8):
    """
    Loads sharded model weights from DeepSpeed FSDP checkpoint files and merges them.
    Converts any DTensors to local tensors before merging.
    """
    merged_state = {}
    shard_files = [
        os.path.join(checkpoint_dir, f"model_world_size_{num_shards}_rank_{rank}.pt")
        for rank in range(num_shards)
    ]
    
    # Temporary storage: key -> list of tensors from each shard
    tmp_storage = {}

    for shard_file in shard_files:
        print(f"Loading shard: {shard_file}")
        shard_state = torch.load(shard_file, map_location="cpu")
        for key, value in shard_state.items():
            if key not in tmp_storage:
                tmp_storage[key] = []
            tmp_storage[key].append(value)

    # Merge shards:
    for key, tensor_list in tmp_storage.items():
        cpu_tensors = []
        for t in tensor_list:
            # If t is a DTensor, convert it to a local tensor.
            if hasattr(t, "to_local"):
                t = t.to_local()
            # Also ensure it is on CPU.
            t = t.cpu().clone().detach()
            cpu_tensors.append(t)
        
        if len(cpu_tensors) == num_shards:
            try:
                merged_state[key] = torch.cat(cpu_tensors, dim=0)
            except Exception as e:
                print(f"Error concatenating key {key}: {e}")
                raise e
        else:
            merged_state[key] = cpu_tensors[0]

    return merged_state


def convert_deepspeed_checkpoint(model_name, checkpoint_dir, output_dir, num_shards=8):
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Merge sharded model state dict.
    print("Merging sharded state dictionaries...")
    merged_state_dict = merge_sharded_state_dict(checkpoint_dir, num_shards)
    
    # Load merged state dict into model.
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    # Save the fully merged model and tokenizer.
    print(f"Saving merged model and tokenizer to {output_dir} ...")
    model.to(torch.bfloat16)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)  # Save tokenizer
    print("Done!")

if __name__ == "__main__":
    original_model = 'pretrained_model/DeepSeek-R1-Distill-Qwen-14B'
    # Adjust these paths as needed.
    checkpoint_directory =  "checkpoints/deepscaler/*/global_step_*/actor"
    output_directory = "Save_Path"

    num_shards = 8  # Number of GPU shards used for training

    convert_deepspeed_checkpoint(original_model, checkpoint_directory, output_directory, num_shards)