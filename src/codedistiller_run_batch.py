# codedistiller_run_batch.py

import json
import os
import re
import time
import asyncio, subprocess
from urllib.parse import urlparse

CONDA_ENV = "codedistiller"
LOG_DIR = "logs"

def env_python(env: str) -> str:
    # Resolve the Python binary for the conda env (once)
    return subprocess.check_output(
        ["conda", "run", "-n", env, "python", "-c", "import sys; print(sys.executable)"],
        text=True,
    ).strip()

def repo_to_logname(arg: str) -> str:
    """
    Turn an arg like:
      - 'https://www.github.com/user/repo'
      - 'https://github.com/user/repo'
      - 'user/repo'
    into: 'logs/log_user_repo.txt'
    """
    s = arg.strip()
    if s.startswith("http"):
        p = urlparse(s)
        path = p.path.strip("/")
    else:
        path = s.strip("/")

    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        user, repo = parts[-2], parts[-1]
    elif len(parts) == 1:
        user, repo = "repo", parts[0]
    else:
        user, repo = "arg", "unknown"

    repo = repo.removesuffix(".git")
    # Add a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    stem = f"log_{user}_{repo}_{timestamp}.txt"
    # make filesystem-safe
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return os.path.join(LOG_DIR, stem)

async def run_one(python_bin: str, arg, config_file:str):
    import random
    # Randomly wait up to 5 seconds before starting
    await asyncio.sleep(random.uniform(0, 5))

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = repo_to_logname(arg)

    # Open once; send both stdout and stderr there
    out = open(log_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            python_bin, "src/codedistiller.py", str(arg), str(config_file),
            stdout=out,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into the same file
        )
        rc = await proc.wait()
        return arg, rc, log_path
    finally:
        out.close()

async def main(args, config_file, limit:int, env=CONDA_ENV):
    pybin = env_python(env)
    sem = asyncio.Semaphore(limit)

    async def worker(a):
        async with sem:
            return await run_one(pybin, a, config_file)

    tasks = [asyncio.create_task(worker(a)) for a in args]
    for fut in asyncio.as_completed(tasks):
        a, rc, log_path = await fut
        print(f"{a} -> exit {rc} (log: {log_path})")



if __name__ == "__main__":
    # Step 0: Add a check for whether the `modal` command line tools are available/visible.  If not, then print a message and exit.
    import os
    import shutil
    # Check whether 'modal list' command is available
    if (not shutil.which("modal")):
        print("The 'modal' command line tools do not appear to be available.  If they are installed, they may not be visible to this batch script (which spawns them in a subprocess) without being called from a terminal, with the conda environment activated (e.g. conda activate myenv, python src/run_batch.py).")
        exit(1)

    # Get the list of repos from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run batch jobs on a list of GitHub repositories.")
    # Two possible arguments: either a JSON file with a list of repos, or a list of repos directly
    parser.add_argument("--repo_file_json", type=str, help="Path to JSON file containing list of repositories.")
    parser.add_argument("--repos_file_txt", type=str, help="Path to text file containing list of repositories, one per line.")
    # Max concurrent jobs
    parser.add_argument("--max_concurrent_jobs", type=int, default=1, help="Maximum number of concurrent jobs to run.")
    # Name of the configuration file
    parser.add_argument("--config_file", type=str, help="Path to the configuration file to use for each repository (e.g. make_example_config.json).")

    args_parsed = parser.parse_args()

    # Set the max concurrent jobs
    MAX_CONCURRENT_JOBS = args_parsed.max_concurrent_jobs

    repo_names = []
    if (args_parsed.repo_file_json):
        print("Loading list of repositories from JSON file:", args_parsed.repo_file_json)
        with open(args_parsed.repo_file_json, "r", encoding="utf-8") as f:
            repo_list = json.load(f)

        for repo in repo_list:
            repo_name = repo["repo_name"]
            if ("www.github.com" not in repo_name) and ("github.com" not in repo_name):
                repo_names.append(f"https://www.github.com/{repo_name}")
            else:
                repo_names.append(repo_name)

    elif (args_parsed.repos_file_txt):
        # Load from text file
        print("Loading list of repositories from text file:", args_parsed.repos_file_txt)
        with open(args_parsed.repos_file_txt, "r", encoding="utf-8") as f:
            for line in f:
                repo_name = line.strip()
                if repo_name:
                    if ("www.github.com" not in repo_name) and ("github.com" not in repo_name):
                        repo_names.append(f"https://www.github.com/{repo_name}")
                    else:
                        repo_names.append(repo_name)

    else:
        print("No input repository list provided. Please specify either --repo_file_json or --repos_file_txt.")
        exit(1)


    print ("Found {} repositories to process.".format(len(repo_names)))
    if (len(repo_names) == 0):
        print("No repositories to process; exiting.")
        exit(0)

    # Show a sample of the repositories to be processed
    num_to_show = 10
    if (len(repo_names) < num_to_show):
        num_to_show = len(repo_names)
    print("Showing first {} repositories:".format(num_to_show))
    for i in range(num_to_show):
        print("  {}".format(repo_names[i]))

    print("Using maximum concurrent jobs: " + str(MAX_CONCURRENT_JOBS))


    # Check if the configuration file is provided
    if (not args_parsed.config_file):
        print("No configuration file provided. Please specify --config_file.")
        exit(1)
    else:
        print("Using configuration file: " + str(args_parsed.config_file))
        # Check that the configuration file exists
        if (not os.path.isfile(args_parsed.config_file)):
            print("Configuration file does not exist: " + str(args_parsed.config_file))
            exit(1)

    print("Pausing for 10 seconds before starting...")
    import time
    time.sleep(10)


    # Start running them
    #asyncio.run(main(args, limit=MAX_CONCURRENT_JOBS))

    if (MAX_CONCURRENT_JOBS < 1):
        print("Maximum concurrent jobs must be at least 1; currently set to: " + str(MAX_CONCURRENT_JOBS))
        exit(1)



    #asyncio.run(main(repo_names, limit=MAX_CONCURRENT_JOBS))
    asyncio.run(main(repo_names, args_parsed.config_file, limit=MAX_CONCURRENT_JOBS))
