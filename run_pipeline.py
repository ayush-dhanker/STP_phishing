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

if __name__ == "__main__":
    run("02_data_pipeline.py")
    run("03_train.py")
    run("04_evaluate.py")
    print("\nPipeline complete.")