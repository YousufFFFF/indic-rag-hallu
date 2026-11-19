import random, sys, time
sys.path.insert(0, "tests/text")
import test_normalize as T
from indichallu.text import normalize as N
total = 0
for code in T.LANGS:
    t = time.perf_counter(); f = 0
    for seed in range(150):
        lang, rng = T.LANGS[code], random.Random(seed)
        vocab = [lang.native_name, lang.digits[:3], lang.native_name[:2], lang.native_name[1:]]
        noise = ["", "", ":", "|", T.ZWNJ, T.ZWJ, T.ZWNJ * 40]
        src = " ".join(rng.choice(noise) + rng.choice(vocab) + rng.choice(noise) for _ in range(60))
        try: T._assert_wordwise_provenance(src, N.indicnlp_normalize(src, code).offsets)
        except AssertionError: f += 1
    total += f
    print(f"{code}: failures {f}/150, {time.perf_counter()-t:.1f}s", flush=True)
print("TOTAL failures:", total, "/ 1350", flush=True)
