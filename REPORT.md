# Model Optimization and Performance Analysis

## Initial Setup and Baseline Performance

Started with inference profiling to identify bottlenecks. Initially attempted to replace SDPA with FlashAttention2, but consistently observed TTFB ≈ 1.64 seconds.

## Profiling Implementation

```python
def run_inference(self, prefill_data: List[str | AudioData]):
    overall_start = time.perf_counter()
    
    # Session creation timing
    session_start = time.perf_counter()
    self.session_id = str(uuid.uuid4())
    session_time = time.perf_counter() - session_start
    print(f"[TIMING] Session creation: {session_time * 1000:.3f}ms")
    
    # Prefill timing
    prefill_start = time.perf_counter()
    if prefill_data:
        self._prefill(data=prefill_data)
    prefill_time = time.perf_counter() - prefill_start
    print(f"[TIMING] Prefill total: {prefill_time * 1000:.3f}ms")
    
    # Generator creation timing
    generator_start = time.perf_counter()
    response_generator = self.model.streaming_generate(...)
    generator_time = time.perf_counter() - generator_start
    print(f"[TIMING] Generator creation: {generator_time * 1000:.3f}ms")
    
    # First response timing (THE CRITICAL MEASUREMENT)
    first_response_start = time.perf_counter()
    for response in response_generator:
        first_response_time = time.perf_counter() - first_response_start
        print(f"[TIMING] Time to first response: {first_response_time * 1000:.3f}ms")
        break
    
    ttfb = time.perf_counter() - overall_start
    print(f"[TIMING] *** TOTAL TTFB: {ttfb * 1000:.3f}ms ***")
```

## Performance Analysis Results

Profiling revealed that ~95% of processing time is consumed by the `streaming_generate` method during model inference. Other operations showed minimal overhead:
- Profile processing: ~3.17% of total time
- GPU transfer: Efficient operations
- AudioData creation (post-processing): Minimal impact

**Key Finding**: The major bottleneck is in the model's forward pass computation, requiring either quantization support, smaller model variants, or hardware acceleration (A100s, H100s).

## Optimization Approaches Tested

### 1. Attention Implementation
- **FlashAttention**: Minimal improvement, TTFB remained 1.6-1.7s, real-time factor ~0.91
- **SDPA**: Better performance when combined with other optimizations

### 2. Quantization Attempts
- **BitsAndBytes**: Issues with GPU allocation due to `init_tts` method duplication
- **Int4 Model**: Installation difficulties with AutoGPTQ, potential accuracy loss concerns

### 3. Batching and Async Operations
- **Batched Prefill**: 
  - Collected all text strings for single `streaming_prefill` call
  - Aggregated audio into single array instead of individual `_prefill_audio` calls
  - Reduced API overhead through batch processing

### 4. JIT Compilation with torch.compile()
- **FlashAttention2 + torch.compile()**: TTFB ~1.5-1.6s, real-time factor ~0.95
- **SDPA + torch.compile()**: Better performance, TTFB ~1.1-1.3s, real-time factor ~0.7-0.75

## Final Optimal Configuration

**Best Performance Achieved**: `torch.compile() + prefill_batching + SDPA`

### Results by Hardware:
- **A10G**: TTFB ~1.1-1.3s, real-time factor ~0.7-0.75
- **H100**: TTFB ~0.84s, real-time factor ~0.54

This configuration delivered sub-1-second TTFB on H100 hardware while maintaining good performance on A10G systems. FEw other observation were, init_tts might be adding to the latency. 