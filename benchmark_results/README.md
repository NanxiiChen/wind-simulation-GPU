# Benchmark Results


## Different number of samples

Fix number of frequencies as 3000, vary number of samples in {10, 25, 50, 100, 200, 500, 1000}.

```bash
python scripts/benchmarks_points.py --backend numpy --max-memory 32.0 --n-frequency 3000
```


## Different frequency

Fix number of points as 100, vary number of frequencies in {100, 500, 1000, 2000, 5000, 8000, 10000}.

```bash
python scripts/benchmarks_freq.py --backend jax --max-memory 6.0 --n-points 1500
```



