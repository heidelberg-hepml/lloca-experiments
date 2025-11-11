import subprocess
from pathlib import Path

import yaml


def run_github_workflow(workflow_file):
    """Parses and executes GitHub Actions workflow steps locally."""
    with open(workflow_file) as f:
        workflow = yaml.safe_load(f)

    print(f"\n📂 Running workflow: {workflow_file}\n")

    for job_name, job in workflow.get("jobs", {}).items():
        print(f"\n🔹 Running job: {job_name}\n")

        for step in job.get("steps", []):
            if "run" in step:
                command = step["run"]
                print(f"➡️ Running: {command}")

                # Execute shell command
                process = subprocess.run(command, shell=True, capture_output=True, text=True)

                # Print output
                if process.stdout:
                    print("✅ Output:\n", process.stdout)
                if process.stderr:
                    print("⚠️ Error:\n", process.stderr)

                if process.returncode != 0:
                    print(f"❌ Step failed: {step.get('name', 'Unnamed step')}")
                    break  # Stop execution on failure


if __name__ == "__main__":
    workflows_dir = Path(".github/workflows")

    # Ensure the workflows folder exists
    if not workflows_dir.exists():
        print("❌ No workflows directory found.")
    else:
        # Loop over all YAML files in the workflows directory
        for workflow_file in workflows_dir.glob("*.yaml"):
            run_github_workflow(workflow_file)
