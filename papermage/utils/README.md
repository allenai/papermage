# utilities

Note from @kylel: Honestly, I kind of hate this; feels like some of these utilities should be baked into `magelib` directly, but not sure where. Will get to it eventually. For now, this README helps keep things a bit more organized.


### merge

Methods that make it easier to combine Entities, which is a fairly common operation when defining new Predictors. For example, many of the Spans of larger Entities like sentences are derived from Token spans merged together.


#### Sep 2023 - Deduping implementations

We had two competing implementations of merging spans. This is just some documentation to help keep track of why we dropped one vs other.

Benchmarking:

```
import random
import time
from papermage.magelib import Span
from papermage.utils.merge import *

# generate random spans, some overlapping some disjoint
many_spans = []
start = 0
for _ in range(10000):
    increment_start = random.choices(population=[-1, 0, 1], weights=[0.2, 0.3, 0.5])
    is_save = random.choices(population=[False, True], weights=[0.8, 0.2])
    if increment_start:
        start += increment_start[0]
    if is_save:
        end = start + random.choices(population=[1, 2, 3, 4, 5],
                                     weights=[0.2, 0.2, 0.2, 0.2, 0.2])[0]
        new_span = Span(start=start, end=end)
        many_spans.append(new_span)
        start = end

# Elapsed: 10.210576057434082
start = time.time()
mcs = MergeClusterSpans(spans=many_spans)
mcs.merge()
end = time.time()
print(f"Elapsed: {end - start}")

# Elapsed: 0.004858255386352539
start = time.time()
results = merge_neighbor_spans(spans=many_spans)
end = time.time()
print(f"Elapsed: {end - start}")
```

MergeClusterSpans is super inefficient. Let's kill it. 