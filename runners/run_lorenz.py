import pandas as pd
from algorithms.lorenz_lyapunov import benchmark_case

cases: list[dict] = [
    dict(name="chaotic_16_45.92_4", sigma=16.0, rho=45.92, beta=4.0),
    dict(name="stable_10_0.5_8_3",  sigma=10.0, rho=0.5,   beta=8.0/3.0)
]

# ======= integration controls =======
t0: float = 0.0
t1: float = 50.0
dt: float = 0.01
transient_time: float = 10.0
observable: str = "x"

# ======= nolds params (tuned to dt) =======
min_tsep: int = int(2.0 / dt)
eck_params: dict = dict(emb_dim=9, matrix_dim=3, tau=10, min_tsep=min_tsep, min_nb=20)
ros_params: dict = dict(emb_dim=9, tau=10, min_tsep=min_tsep, trajectory_len=60, fit="RANSAC")

# ======= run =======
all_rows: list[dict] = []
for case in cases:
    rows = benchmark_case(case, t0, t1, dt, transient_time, eck_params, ros_params, observable)
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df = df[["case","method","time_sec","lce1","lce2","lce3","sigma","rho","beta"]]
df = df.sort_values(["case","method"]).reset_index(drop=True)
print(df.to_string(index=False))
