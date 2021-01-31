import subprocess
import time
from tqdm import tqdm

from core.data import list_particles

def main():
    processes = []
    commands = []
    for particle in list_particles:
        for num_layers in [2, 5, 10, 20]:
            commands.append(
                f"sbatch -t 15000 --gpus=1 -p normal -c 4 run_one_job.sh "
                f"data.data_dir=/home/martemev/Datasets/RICH-2021/ "
                f"experiment.particle={particle} "
                f"model.C.num_layers={num_layers} model.G.num_layers={num_layers}"
            )

    batch_size = 20
    for command in tqdm(commands):
        print(command)
        process = subprocess.Popen(command,
                                   shell=True,
                                   close_fds=True,
                                   )
        processes.append(process)
        pr_count = subprocess.Popen("squeue | grep martemev | wc -l", shell=True, stdout=subprocess.PIPE)
        out, err = pr_count.communicate()
        if int(out) > batch_size:
            while int(out) > batch_size:
                print("Waiting... ")
                time.sleep(240)
                pr_count = subprocess.Popen("squeue | grep martemev | wc -l", shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()

    for process in processes:
        print(process.pid)


if __name__ == "__main__":
    main()
