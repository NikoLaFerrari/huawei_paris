import shutil
import os


def generate_copy(input_file, suffix, new_config):
    """ "config copy"""
    if not os.path.isfile(input_file):
        print(f"The file '{input_file}' does not exist.")
        return

    base_name, ext = os.path.splitext(input_file)
    base_name = os.path.basename(base_name)
    new_file_name = (
        "./toolkit/perf_estimation/last_configs/"
        + f"{base_name}_{suffix}{ext}"
    )

    shutil.copy(input_file, new_file_name)
    print(f"Created copy: {new_file_name}")

    new_content = []

    with open(new_file_name, "r") as file:
        content = file.readlines()
        skip_lines = 0
        for line in content:
            if skip_lines:
                skip_lines -= 1
                continue
            if "epochs:" in line:
                new_content.append(line)
                skip_lines = 1
                new_content.append(
                    f"  batch_size: {new_config['batch_size']}\n"
                )
            elif "parallel_config:" in line:
                new_content.append(line)
                skip_lines = 5
                new_content.append(
                    f"  data_parallel: {new_config['data_parallel']}\n"
                )
                new_content.append(
                    f"  model_parallel: {new_config['model_parallel']}\n"
                )
                new_content.append(
                    f"  pipeline_stage: {new_config['pipeline_stage']}\n"
                )
                new_content.append(
                    f"  use_seq_parallel: {new_config['use_seq_parallel']}\n"
                )
                new_content.append(
                    f"  micro_batch_num: {new_config['micro_batch_num']}\n"
                )
            elif "do_sample:" in line:
                new_content.append(line)
                skip_lines = 1
                new_content.append(
                    f"    pp_interleave_num: "
                    f"{new_config['pp_interleave_num']}\n"
                )
            else:
                new_content.append(line)

    with open(new_file_name, "w") as file:
        file.writelines(new_content)

    print(f"Replaced 'parallel_config' in {new_file_name}.")


def valid_config(d, t, p, vp, b, m, N, L, fix_gbs):
    """strat verification"""
    return (
        d * t * p == N
        and p <= m
        and (vp == 1 or p != 1)
        and vp <= L / p
        and (not fix_gbs or d * b * (m if p != 1 else 1) == fix_gbs)
    )


def generate(
    input_file="./llama3_70b.yaml",
    config_list_file=None,
    fix_global_batch=None,
    nb_devices=64,
):
    """generate config yamls"""
    N = nb_devices
    L = N
    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "num_layers:" in line:
                L = int(list(line.split())[-1])
                break

    X = [1]
    while 2 * X[-1] <= N:
        X.append(2 * X[-1])

    configs = []
    if not config_list_file:
        for d in X:
            for t in X:
                for p in X:
                    for m in X:
                        for b in X:
                            for vp in X:
                                if valid_config(
                                    d, t, p, vp, b, m, N, L, fix_global_batch
                                ):
                                    configs.append([d, t, p, vp, b, m])
    else:
        with open(config_list_file, "r") as ff:
            lines = ff.readlines()
            for line in lines:
                values = list(map(int, line.split()))
                if len(values) == 6:
                    d, t, p, vp, b, m = values
                    if valid_config(d, t, p, vp, b, m, N, L, fix_global_batch):
                        configs.append([d, t, p, vp, b, m])

    print(f"Generating {len(configs)} yamls..")
    with open(
        "./toolkit/perf_estimation/last_configs/config_list.txt", "w"
    ) as f:
        for [d, t, p, vp, b, m] in configs:
            f.write(f"{d} {t} {p} {vp} {b} {m}\n")
            new_parallel_config = {
                "data_parallel": d,
                "model_parallel": t,
                "pipeline_stage": p,
                "use_seq_parallel": False,
                "micro_batch_num": m,
                "batch_size": b,
                "pp_interleave_num": vp,
            }
            suffix = f"{d}_{t}_{p}_{vp}_{b}_{m}"
            generate_copy(input_file, suffix, new_parallel_config)
