# LaViC repo (cloned)

Note that the problems are all related to `src/prompt_tuning.py` file.
## The problems
The problems are the following:
According to the LaViC paper, the source code is intended to execute **in a single GPU**. However, our gpu has less VRAM than the one stated in the paper. The solution is to make 4-bit quantization, which hurts the benchmarked performance.

```python
base_model = LlavaForConditionalGeneration.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True
).to(device)
```


## The (not worked) solution
First, we identified that the fitting step is based on pytorch-lightning library and it supports offloading the model to multiple GPU. The method called **fully sharded data parallel (FSDP)**, which can be easily used by setting the trainer to.

```python
trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    accelerator="gpu",
    devices=-1,
    strategy=FSDPStrategy(
        cpu_offload=True,
    ),
    callbacks=[],
    precision='16',
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)
```

However, as soon as we set this. There is a data type mismatch (which I believe it is the mismatch between llava model and LoRA, where one is torch.float16 and another is torch.float32).

<details>
<summary>The error</summary>
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/LaViC/src/prompt_tuning.py", line 729, in <module>
[rank0]:     main()
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/LaViC/src/prompt_tuning.py", line 647, in main
[rank0]:     trainer.fit(llava_model, train_loader, val_loader)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 561, in fit
[rank0]:     call._call_and_handle_interrupt(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 47, in _call_and_handle_interrupt
[rank0]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank0]:     return function(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 599, in _fit_impl
[rank0]:     self._run(model, ckpt_path=ckpt_path)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 988, in _run
[rank0]:     self.strategy.setup(self)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/fsdp.py", line 341, in setup
[rank0]:     self.model = self._setup_model(self.model)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/fsdp.py", line 307, in _setup_model
[rank0]:     model = FullyShardedDataParallel(
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 499, in __init__
[rank0]:     _init_param_handle_from_module(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_init_utils.py", line 622, in _init_param_handle_from_module
[rank0]:     _init_param_handle_from_params(state, managed_params, fully_sharded_module)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_init_utils.py", line 634, in _init_param_handle_from_params
[rank0]:     handle = FlatParamHandle(
[rank0]:              ^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 588, in __init__
[rank0]:     self._init_flat_param_and_metadata(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 641, in _init_flat_param_and_metadata
[rank0]:     ) = self._validate_tensors_to_flatten(params)
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 785, in _validate_tensors_to_flatten
[rank0]:     raise ValueError(
[rank0]: ValueError: Must flatten tensors with uniform dtype but got torch.float16 and torch.float32
```
</details>

So, I typecasted the model into torch.float16 (in `src/prompt_tuning.py line 630`).
```python
llava_model = llava_model.to(torch.bfloat16)
```

However, the OOM errors strike again...
<details>
<summary>The error</summary>
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/LaViC/src/prompt_tuning.py", line 729, in <module>
[rank0]:     main()
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/LaViC/src/prompt_tuning.py", line 647, in main
[rank0]:     trainer.fit(llava_model, train_loader, val_loader)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 561, in fit
[rank0]:     call._call_and_handle_interrupt(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 47, in _call_and_handle_interrupt
[rank0]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank0]:     return function(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 599, in _fit_impl
[rank0]:     self._run(model, ckpt_path=ckpt_path)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 988, in _run
[rank0]:     self.strategy.setup(self)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/fsdp.py", line 341, in setup
[rank0]:     self.model = self._setup_model(self.model)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/pytorch_lightning/strategies/fsdp.py", line 307, in _setup_model
[rank0]:     model = FullyShardedDataParallel(
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 499, in __init__
[rank0]:     _init_param_handle_from_module(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_init_utils.py", line 622, in _init_param_handle_from_module
[rank0]:     _init_param_handle_from_params(state, managed_params, fully_sharded_module)
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_init_utils.py", line 634, in _init_param_handle_from_params
[rank0]:     handle = FlatParamHandle(
[rank0]:              ^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 588, in __init__
[rank0]:     self._init_flat_param_and_metadata(
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 750, in _init_flat_param_and_metadata
[rank0]:     self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
[rank0]:                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 872, in flatten_tensors_into_flat_param
[rank0]:     flat_param_data = self.flatten_tensors(tensors, aligned_numel)
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/research/d7/fyp24/tpipatpajong2/miniconda3/envs/fyp/lib/python3.12/site-packages/torch/distributed/fsdp/_flat_param.py", line 864, in flatten_tensors
[rank0]:     return torch.cat(flat_tensors, dim=0)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.14 GiB. GPU 0 has a total capacity of 23.60 GiB of which 8.22 GiB is free. Including non-PyTorch memory, this process has 15.36 GiB memory in use. Of the allocated memory 14.14 GiB is allocated by PyTorch, and 499.78 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```
</details>