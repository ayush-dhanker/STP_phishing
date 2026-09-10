import subprocess
import sys


def run(script):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Pipeline stopped.")
        sys.exit(1)


def run_feast_apply():
    print(f"\n{'='*60}")
    print("Running: feast apply (feature_repo/)")
    print('='*60)
    result = subprocess.run(["feast", "apply"], cwd="feature_repo")
    if result.returncode != 0:
        print("\nERROR: feast apply failed. Pipeline stopped.")
        sys.exit(1)


if __name__ == "__main__":
    run("pipeline_feast.py")
    run_feast_apply()
    run("03_train_kfold.py")
    run("04_evaluate.py")
    print("\nPipeline complete.")