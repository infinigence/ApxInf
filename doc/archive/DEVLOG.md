# Development log

## Performance History

Add Batch Command Buffer (157b0e0)

```
=== Generation Profile ===
Input tokens:     25
Output tokens:    16
TTFT:             1159.4 ms
TPOT:             77.8 ms/token
Generation TPS:   12.86 tok/s
Total latency:    2.33 s
=========================
```

Put all kernels on Metal Backend

```
=== Generation Profile ===
Input tokens:     25
Output tokens:    16
TTFT:             1668.3 ms
TPOT:             151.9 ms/token
Generation TPS:   6.58 tok/s
Total latency:    3.95 s
=========================
```

```
=== Generation Profile ===
Input tokens:     25
Output tokens:    16
TTFT:             1279.8 ms
TPOT:             201.0 ms/token
Generation TPS:   4.98 tok/s
Total latency:    4.29 s
=========================
```